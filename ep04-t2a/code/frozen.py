"""冻结的现成零件:都是公开下载的预训练模型,训练前把数据过一遍、缓存到磁盘,训练时不再跑。

生成用(压缩成潜空间,生成完再还原):
  ImageVAE  FLUX.2 VAE           64x64x3        -> 32 x 8 x 8
  VideoVAE  Wan2.1 VAE(时空)     16 帧 64x64x3  -> 16 x 5 x 8 x 8(时间压 4 倍、空间压 8 倍)
  Voice     Kyutai Mimi          2.08 秒 24 kHz -> 26 帧 x 512(每秒 12.5 帧,取不量化的连续潜变量)
理解用(只编码不还原,经小投影层接进主干):
  See       SigLIP2-base 视觉塔  一张图 -> 4x4=16 个 768 维向量;视频每帧 2x2=4 个
  Hear      Whisper-tiny 编码器  2.08 秒 -> 26 个 384 维向量
"""
import torch
import torch.nn.functional as F

from parts import AUD_FRAMES, CLIP, SR, to_float as u8_to_pm1, to_u8 as pm1_to_u8

REPOS = {
    "image_vae": ("black-forest-labs/FLUX.2-klein-4B", "vae"),
    "video_vae": ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "vae"),
    "voice": ("kyutai/mimi", None),
    "see": ("google/siglip2-base-patch16-224", None),
    "hear": ("openai/whisper-tiny", None),
}
dev = "cuda" if torch.cuda.is_available() else "cpu"   # 没有卡时在 CPU 上跑(只用于冒烟)


class ImageVAE:
    def __init__(self):
        from diffusers import AutoencoderKLFlux2
        repo, sub = REPOS["image_vae"]
        self.m = AutoencoderKLFlux2.from_pretrained(repo, subfolder=sub).to(dev).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, u8):                           # [B, 64, 64, 3] -> [B, 32, 8, 8]
        return self.m.encode(u8_to_pm1(u8)).latent_dist.mode()

    @torch.no_grad()
    def decode(self, z):
        return pm1_to_u8(self.m.decode(z).sample)


class VideoVAE:
    """Wan 的时空 VAE 是因果的:第 1 帧单独压,之后每 4 帧压成 1 帧,所以要 1+4k 帧。16 帧末尾补 1 帧成 17 帧。"""

    def __init__(self):
        from diffusers import AutoencoderKLWan
        repo, sub = REPOS["video_vae"]
        self.m = AutoencoderKLWan.from_pretrained(repo, subfolder=sub).to(dev).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, u8):                           # [B, 16, 64, 64, 3] -> [B, 16, 5, 8, 8]
        x = u8_to_pm1(u8).transpose(1, 2)           # [B, 3, T, H, W]
        x = torch.cat([x, x[:, :, -1:]], 2)
        return self.m.encode(x).latent_dist.mode()

    @torch.no_grad()
    def decode(self, z):                            # -> [B, 16, 64, 64, 3]
        return pm1_to_u8(self.m.decode(z).sample[:, :, :16].transpose(1, 2))


class Voice:
    def __init__(self):
        from transformers import MimiModel
        self.m = MimiModel.from_pretrained(REPOS["voice"][0]).to(dev).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, wav):                          # [B, CLIP] -> [B, 26, 512]
        """取「不量化」的那条路:编码器输出只过量化器的输入/输出投影,跳过码本查表。
        Mimi 训练时有一半时间就这样绕过量化,所以解码器认得这种连续向量。"""
        m = self.m
        e = m.encoder(wav[:, None].float())
        e = m.encoder_transformer(e.transpose(1, 2))[0].transpose(1, 2)
        e = m.downsample(e)
        q = m.quantizer
        sem, ac = q.semantic_residual_vector_quantizer, q.acoustic_residual_vector_quantizer
        return (sem.output_proj(sem.input_proj(e)) + ac.output_proj(ac.input_proj(e))).transpose(1, 2)

    @torch.no_grad()
    def decode(self, z):                            # [B, 26, 512] -> [B, CLIP]
        m = self.m
        e = m.upsample(z.transpose(1, 2).float())
        e = m.decoder_transformer(e.transpose(1, 2))[0].transpose(1, 2)
        return m.decoder(e)[:, 0, :CLIP]


class See:
    def __init__(self):
        from transformers import SiglipVisionModel
        self.m = SiglipVisionModel.from_pretrained(REPOS["see"][0]).to(dev).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, u8, grid):                     # [B, 64, 64, 3] -> [B, grid*grid, 768]
        x = F.interpolate(u8_to_pm1(u8), size=224, mode="bicubic", align_corners=False)   # SigLIP 的归一化正是 [-1,1]
        h = self.m(pixel_values=x).last_hidden_state                                     # [B, 196, 768]
        B = h.shape[0]
        h = h.transpose(1, 2).reshape(B, -1, 14, 14)
        return F.adaptive_avg_pool2d(h, grid).flatten(2).transpose(1, 2)


class Hear:
    def __init__(self):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.fx = WhisperFeatureExtractor.from_pretrained(REPOS["hear"][0])
        self.m = WhisperModel.from_pretrained(REPOS["hear"][0]).encoder.to(dev).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, wav):                          # [B, CLIP] @24k -> [B, 26, 384]
        import librosa
        w16 = [librosa.resample(w, orig_sr=SR, target_sr=16000) for w in wav.float().cpu().numpy()]
        feats = self.fx(w16, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
        h = self.m(feats).last_hidden_state[:, :AUD_FRAMES * 4]      # Whisper 每 20ms 一帧,2.08 秒 = 104 帧
        return F.avg_pool1d(h.transpose(1, 2), 4).transpose(1, 2)    # 每 4 帧平均(80ms)-> 26 个,和 Mimi 对齐


def params_M(obj):
    return round(sum(p.numel() for p in obj.m.parameters()) / 1e6, 1)
