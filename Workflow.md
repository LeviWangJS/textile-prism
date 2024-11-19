graph TB
    subgraph 初始化阶段
        A[加载配置] --> B[初始化日志]
        B --> C[设置设备]
        C --> D[创建目录]
    end

    subgraph 模型初始化
        D --> E[初始化PatternTransformer]
        E --> F[CLIP特征提取器]
        E --> G[原始特征提取器]
        E --> H[特征融合模块]
        F & G & H --> I[编码器解码器]
    end

    subgraph 数据准备
        D --> J[创建数据集]
        J --> K[训练集DataLoader]
        J --> L[验证集DataLoader]
    end

    subgraph 训练组件
        D --> M[初始化优化器]
        D --> N[初始化损失函数]
        D --> O[学习率调度器]
        D --> P[早停机制]
    end

    subgraph 监控系统
        D --> Q[SmartTrainingMonitor]
        Q --> R[OpenAIHelper]
        Q --> S[性能分析]
        Q --> T[优化建议]
    end

    subgraph 可视化系统
        D --> U[TrainingVisualizer]
        D --> V[FeatureVisualizer]
        U --> W[损失曲线]
        V --> X[特征分布]
        W & X --> Y[ReportGenerator]
    end

    subgraph 训练循环
        Z[开始训练] --> AA[批次训练]
        AA --> AB[前向传播]
        AB --> AC[损失计算]
        AC --> AD[反向传播]
        AD --> AE[参数更新]
        AE --> AF[批次监控]
        AF --> AG[特征可视化]
        AG --> AH[验证评估]
        AH --> AI[更新学习率]
        AI --> AJ[检查早停]
        AJ --> |继续训练| AA
        AJ --> |满足条件| AK[结束训练]
    end

    subgraph 训练输出
        AK --> AL[保存最佳模型]
        AK --> AM[生成训练报告]
        AK --> AN[保存可视化结果]
    end

    %% 连接主要流程
    I --> Z
    K & L --> Z
    M & N & O & P --> Z
    Q --> AF
    U & V --> AG

    %% 样式设置
    classDef init fill:#f9f,stroke:#333,stroke-width:2px
    classDef model fill:#bbf,stroke:#333,stroke-width:2px
    classDef data fill:#bfb,stroke:#333,stroke-width:2px
    classDef train fill:#fbf,stroke:#333,stroke-width:2px
    classDef monitor fill:#ffb,stroke:#333,stroke-width:2px
    classDef visual fill:#bff,stroke:#333,stroke-width:2px
    classDef output fill:#fbb,stroke:#333,stroke-width:2px

    class A,B,C,D init
    class E,F,G,H,I model
    class J,K,L data
    class M,N,O,P,Z,AA,AB,AC,AD,AE train
    class Q,R,S,T,AF monitor
    class U,V,W,X,Y,AG visual
    class AK,AL,AM,AN output