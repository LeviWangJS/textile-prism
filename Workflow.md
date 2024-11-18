graph TD
    A[数据加载] --> B[训练循环]
    B --> C[批次训练]
    C --> D[损失计算]
    D --> E[SmartTrainingMonitor]
    
    subgraph 监控分析
        E --> F[批次分析]
        E --> G[Epoch分析]
        F --> H[OpenAIHelper]
        G --> H
        H --> I[生成建议]
    end
    
    subgraph 指标分析
        D --> J[TrainingAnalyzer]
        J --> K[损失趋势分析]
        J --> L[统计指标计算]
    end
    
    subgraph 可视化报告
        K --> M[TrainingVisualizer]
        L --> N[ReportGenerator]
        M --> O[损失曲线]
        N --> P[训练报告]
    end
    
    I --> B
    O --> P