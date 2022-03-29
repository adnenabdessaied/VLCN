# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import SA, MCA_ED, VLC
from core.model.dnc import DNC

import torch.nn as nn
import torch.nn.functional as F
import torch

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class AttFlatMem(AttFlat):
    def __init__(self, __C):
        super(AttFlatMem, self).__init__(__C)
        self.__C = __C

    def forward(self, x_mem, x, x_mask):
        att = self.mlp(x_mem)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            float('-inf')
        )
        att = F.softmax(att, dim=1)
        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted
# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net1(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net1, self).__init__()
        print('Training with Network type 1: VLCN')
        self.pretrained_path = __C.PRETRAINED_PATH
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.frame_feat_linear = nn.Linear(
            __C.FRAME_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.clip_feat_linear = nn.Linear(
            __C.CLIP_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )
        self.backbone = VLC(__C)

        self.attflat_lang = AttFlat(__C)
        self.attflat_frame = AttFlat(__C)
        self.attflat_clip = AttFlat(__C)

        self.dnc = DNC(
            __C.FLAT_OUT_SIZE,
            __C.FLAT_OUT_SIZE,
            rnn_type='lstm',
            num_layers=2,
            num_hidden_layers=2,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
            nr_cells=__C.CELL_COUNT_DNC,
            read_heads=__C.N_READ_HEADS_DNC,
            cell_size=__C.WORD_LENGTH_DNC,
            nonlinearity='tanh',
            gpu_id=0,
            independent_linears=False,
            share_memory=False,
            debug=False,
            clip=20,
        )

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)

        self.proj_norm_dnc = LayerNorm(__C.FLAT_OUT_SIZE + __C.N_READ_HEADS_DNC * __C.WORD_LENGTH_DNC)
        self.linear_dnc = FC(__C.FLAT_OUT_SIZE + __C.N_READ_HEADS_DNC * __C.WORD_LENGTH_DNC, __C.FLAT_OUT_SIZE, dropout_r=0.2)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def forward(self, frame_feat, clip_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        frame_feat_mask = self.make_mask(frame_feat)
        clip_feat_mask = self.make_mask(clip_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)


        # Pre-process Video Feature
        frame_feat = self.frame_feat_linear(frame_feat)
        clip_feat = self.clip_feat_linear(clip_feat)

        # Backbone Framework
        lang_feat, frame_feat, clip_feat = self.backbone(
            lang_feat,
            frame_feat,
            clip_feat,
            lang_feat_mask,
            frame_feat_mask,
            clip_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        frame_feat = self.attflat_frame(
            frame_feat,
            frame_feat_mask
        )

        clip_feat = self.attflat_clip(
            clip_feat,
            clip_feat_mask
        )
        proj_feat_0 = lang_feat + frame_feat + clip_feat
        proj_feat_0 = self.proj_norm(proj_feat_0)

        proj_feat_1 = torch.stack([lang_feat, frame_feat, clip_feat], dim=1)
        proj_feat_1, (_, _, rv), _ = self.dnc(proj_feat_1, (None, None, None), reset_experience=True, pass_through_memory=True)
        proj_feat_1 = proj_feat_1.sum(1)
        proj_feat_1 = torch.cat([proj_feat_1, rv], dim=-1)
        proj_feat_1 = self.proj_norm_dnc(proj_feat_1)
        proj_feat_1 = self.linear_dnc(proj_feat_1)
        # proj_feat_1 = self.proj_norm(proj_feat_1)

        proj_feat = torch.sigmoid(self.proj(proj_feat_0 + proj_feat_1))

        return proj_feat

    def load_pretrained_weights(self):
        pretrained_msvd = torch.load(self.pretrained_path)['state_dict']
        for n_pretrained, p_pretrained in pretrained_msvd.items():
            if 'dnc' in n_pretrained:
                self.state_dict()[n_pretrained].copy_(p_pretrained)
        print('Pre-trained dnc-weights successfully loaded!')
    
    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

class Net2(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net2, self).__init__()
        print('Training with Network type 2: VLCN-FLF')
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )
        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.frame_feat_linear = nn.Linear(
            __C.FRAME_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.clip_feat_linear = nn.Linear(
            __C.CLIP_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )
        self.backbone = VLC(__C)

        self.attflat_lang = AttFlat(__C)
        self.attflat_frame = AttFlat(__C)
        self.attflat_clip = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, frame_feat, clip_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        frame_feat_mask = self.make_mask(frame_feat)
        clip_feat_mask = self.make_mask(clip_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)


        # Pre-process Video Feature
        frame_feat = self.frame_feat_linear(frame_feat)
        clip_feat = self.clip_feat_linear(clip_feat)

        # Backbone Framework
        lang_feat, frame_feat, clip_feat = self.backbone(
            lang_feat,
            frame_feat,
            clip_feat,
            lang_feat_mask,
            frame_feat_mask,
            clip_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        frame_feat = self.attflat_frame(
            frame_feat,
            frame_feat_mask
        )

        clip_feat = self.attflat_clip(
            clip_feat,
            clip_feat_mask
        )
        proj_feat = lang_feat + frame_feat + clip_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat
    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

class Net3(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net3, self).__init__()
        print('Training with Network type 3: VLCN+LSTM')

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.frame_feat_linear = nn.Linear(
            __C.FRAME_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.clip_feat_linear = nn.Linear(
            __C.CLIP_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )
        self.backbone = VLC(__C)

        self.attflat_lang = AttFlat(__C)
        self.attflat_frame = AttFlat(__C)
        self.attflat_clip = AttFlat(__C)

        self.lstm_fusion = nn.LSTM(
            input_size=__C.FLAT_OUT_SIZE,
            hidden_size=__C.FLAT_OUT_SIZE,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj_feat_1 = nn.Linear(__C.FLAT_OUT_SIZE * 2, __C.FLAT_OUT_SIZE)

        self.proj_norm_lstm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def forward(self, frame_feat, clip_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        frame_feat_mask = self.make_mask(frame_feat)
        clip_feat_mask = self.make_mask(clip_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)


        # Pre-process Video Feature
        frame_feat = self.frame_feat_linear(frame_feat)
        clip_feat = self.clip_feat_linear(clip_feat)

        # Backbone Framework
        lang_feat, frame_feat, clip_feat = self.backbone(
            lang_feat,
            frame_feat,
            clip_feat,
            lang_feat_mask,
            frame_feat_mask,
            clip_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        frame_feat = self.attflat_frame(
            frame_feat,
            frame_feat_mask
        )

        clip_feat = self.attflat_clip(
            clip_feat,
            clip_feat_mask
        )
        proj_feat_0 = lang_feat + frame_feat + clip_feat
        proj_feat_0 = self.proj_norm(proj_feat_0)

        proj_feat_1 = torch.stack([lang_feat, frame_feat, clip_feat], dim=1)
        proj_feat_1, _ = self.lstm_fusion(proj_feat_1)
        proj_feat_1 = proj_feat_1.sum(1)
        proj_feat_1 = self.proj_feat_1(proj_feat_1)
        proj_feat_1 = self.proj_norm_lstm(proj_feat_1)

        proj_feat = torch.sigmoid(self.proj(proj_feat_0 + proj_feat_1))

        return proj_feat

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

class Net4(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net4, self).__init__()
        print('Training with Network type 4: MCAN')
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.frame_feat_linear = nn.Linear(
            __C.FRAME_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.clip_feat_linear = nn.Linear(
            __C.CLIP_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )
        self.backbone = MCA_ED(__C)

        self.attflat_lang = AttFlat(__C)
        self.attflat_vid = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, frame_feat, clip_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        frame_feat_mask = self.make_mask(frame_feat)
        clip_feat_mask = self.make_mask(clip_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)


        # Pre-process Video Feature
        frame_feat = self.frame_feat_linear(frame_feat)
        clip_feat = self.clip_feat_linear(clip_feat)

        # concat frame and clip features
        vid_feat = torch.cat([frame_feat, clip_feat], dim=1)
        vid_feat_mask = torch.cat([frame_feat_mask, clip_feat_mask], dim=-1)
        # Backbone Framework
        lang_feat, vid_feat = self.backbone(
            lang_feat,
            vid_feat,
            lang_feat_mask,
            vid_feat_mask,
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        vid_feat = self.attflat_vid(
            vid_feat,
            vid_feat_mask
        )

        proj_feat = lang_feat + vid_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


