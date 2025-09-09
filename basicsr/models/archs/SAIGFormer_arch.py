import torch
import torch.nn as nn
import basicsr.models.archs.transformer_block as transformer
from basicsr.models.archs.SAI2E import SAI2E

class SAIGFormer(nn.Module):
    def __init__(self, embed_dim=32, k_s=3, encoder_num_blocks=[4,6,6,8], decoder_num_blocks=[6,6,4, 4], ffn_expansion_factor=2.66, heads=[1,2,4,8], train_patch=128, eps=0):

        super(SAIGFormer, self).__init__()

        # Transformer Encoder and decoder
        inp_channels = 3
        out_channels=3
        bias = False

        # encoder
        self.patch_embed = transformer.OverlapPatchEmbed(inp_channels, embed_dim, bias=False)

        self.svp = SAI2E(in_channels=inp_channels, train_patch=train_patch, eps=eps)
        self.encoder_level1 = nn.ModuleList([transformer.SAIGTransformer(dim=embed_dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(encoder_num_blocks[0])])

        self.svp_down1_2 = nn.Conv2d(inp_channels, inp_channels, 4, 2, 1, bias=bias, groups=inp_channels)
        self.down1_2 = transformer.Downsample(embed_dim)
        self.encoder_level2 = nn.ModuleList([transformer.SAIGTransformer(dim=int(embed_dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(encoder_num_blocks[1])])

        self.svp_down2_3 = nn.Conv2d(inp_channels, inp_channels, 4, 2, 1, bias=bias, groups=inp_channels)
        self.down2_3 = transformer.Downsample(int(embed_dim*2**1))
        self.encoder_level3 = nn.ModuleList([transformer.SAIGTransformer(dim=int(embed_dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(encoder_num_blocks[2])])

        self.svp_down3_4 = nn.Conv2d(inp_channels, inp_channels, 4, 2, 1, bias=bias, groups=inp_channels)
        self.down3_4 = transformer.Downsample(int(embed_dim*2**2))
        self.latent = nn.ModuleList([transformer.SAIGTransformer(dim=int(embed_dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(encoder_num_blocks[3])])

        # decoder
        self.decoder_latent = nn.ModuleList([transformer.SAIGTransformer(dim=int(embed_dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(1)])

        self.up4_3 = transformer.Upsample(int(embed_dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(embed_dim*2**3), int(embed_dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([transformer.SAIGTransformer(dim=int(embed_dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(decoder_num_blocks[0])])


        self.up3_2 = transformer.Upsample(int(embed_dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(embed_dim*2**2), int(embed_dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([transformer.SAIGTransformer(dim=int(embed_dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(decoder_num_blocks[1])])

        self.up2_1 = transformer.Upsample(int(embed_dim*2**1))
        self.decoder_level1 = nn.ModuleList([transformer.SAIGTransformer(dim=int(embed_dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(decoder_num_blocks[2])])


        # refinement
        self.refinement = nn.ModuleList([transformer.SAIGTransformer(dim=int(embed_dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(decoder_num_blocks[-1])])

        self.output = nn.Conv2d(int(embed_dim*2**1), out_channels, kernel_size=k_s, stride=1, padding=k_s//2, bias=bias)

    def forward(self, inp_img):
        svp_img_1 = self.svp(inp_img)

        # encoder
        inp_enc_level1 = self.patch_embed(inp_img)
        for SAIGTransformer_block in self.encoder_level1:
            inp_enc_level1 = SAIGTransformer_block((inp_enc_level1, svp_img_1))
        out_enc_level1 = inp_enc_level1


        inp_enc_level2 = self.down1_2(out_enc_level1)
        svp_img_2 = self.svp_down1_2(svp_img_1)
        for SAIGTransformer_block in self.encoder_level2: 
            inp_enc_level2 = SAIGTransformer_block((inp_enc_level2, svp_img_2))
        out_enc_level2 = inp_enc_level2

        inp_enc_level3 = self.down2_3(out_enc_level2)
        svp_img_3 = self.svp_down2_3(svp_img_2)
        for SAIGTransformer_block in self.encoder_level3: 
            inp_enc_level3 = SAIGTransformer_block((inp_enc_level3, svp_img_3))
        out_enc_level3 = inp_enc_level3

        inp_enc_level4 = self.down3_4(out_enc_level3)
        svp_img_4 = self.svp_down3_4(svp_img_3)
        for SAIGTransformer_block in self.latent: 
            inp_enc_level4 = SAIGTransformer_block((inp_enc_level4, svp_img_4))
        latent = inp_enc_level4

        # decoder
        for SAIGTransformer_block in self.decoder_latent: 
            latent = SAIGTransformer_block((latent, svp_img_4))
        inp_dec_level3 = latent
        inp_dec_level3 = self.up4_3(inp_dec_level3)


        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        for SAIGTransformer_block in self.decoder_level3: 
            inp_dec_level3 = SAIGTransformer_block((inp_dec_level3, svp_img_3))
        out_dec_level3 = inp_dec_level3

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        for SAIGTransformer_block in self.decoder_level2: 
            inp_dec_level2 = SAIGTransformer_block((inp_dec_level2, svp_img_2))
        out_dec_level2 = inp_dec_level2

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        for SAIGTransformer_block in self.decoder_level1: 
            inp_dec_level1 = SAIGTransformer_block((inp_dec_level1, svp_img_1)) 
        out_dec_level1 = inp_dec_level1


        for SAIGTransformer_block in self.refinement: 
            out_dec_level1 = SAIGTransformer_block((out_dec_level1, svp_img_1)) 

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


