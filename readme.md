# DataFlow

## 简介

探索数据处理流程，包括数据提取、数据清洗等环节。

## 数据处理流程

```mermaid
    flowchart LR
    subgraph 源数据
    Doc[文档]
    Mes[聊天记录]
    end

    subgraph 提取 QA
    Doc_QA[document.py]
    Mes_QA[message.py]
    end
    Doc --> Doc_QA
    Mes --> Mes_QA

    Doc_QA --> Clear[数据清洗]
    Mes_QA --> Clear
    Clear --> DB[数据库]
```

## 进度


## 待完成
- [ ] 文件处理过程全面转向数据库
- [ ] 整理数据流程，构建完整的数据处理流程
- [ ] 使用 MongoDB 存储数据