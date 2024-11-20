class PatternTransformer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        # CLIP特征提取器
        self.clip_extractor = CLIPFeatureExtractor(config['feature_extractor'])
        
        # 原始特征提取器
        self.original_extractor = OriginalFeatureExtractor(config)
        
        # 特征融合
        self.feature_fusion = FeatureFusion(config['feature_fusion'])
        
        # 编码器和解码器
        self.encoder = TransformerEncoder(config['encoder'])
        self.decoder = TransformerDecoder(config['decoder'])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提取CLIP特征
        clip_features = self.clip_extractor.extract_features(x)
        
        # 提取原始特征
        original_features = self.original_extractor(x)
        
        # 特征融合
        fused_features = self.feature_fusion(clip_features, original_features)
        
        # 编码和解码
        encoded = self.encoder(fused_features)
        output = self.decoder(encoded)
        
        return output 