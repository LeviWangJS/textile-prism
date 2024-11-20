def update_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """更新DataLoader的批次大小"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )
