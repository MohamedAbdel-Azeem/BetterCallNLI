# NDA Review — Multi-Agent Architecture (Updated)

```mermaid
graph TD
    %% Node Definitions
    Start([Contract + user prompt<br/>+ conversation history])
    ConvAgent(Conversation agent<br/>Routes request, manages history)
    
    subgraph Loop ["For each hypothesis (x17)"]
        direction TB
        Router{Retrieval router<br/>CLI flag: vector RAG or GraphRAG}
        
        subgraph Retrieval_Logic ["Retrieval Pipeline"]
            Vector[Vector RAG pipeline<br/>Embedding + cosine similarity]
            Graph[GraphRAG pipeline<br/>Knowledge graph retrieval]
            Chroma[(Chroma<br/>vector DB)]
            Neo4j[(Neo4j<br/>graph DB)]
            Context[Retrieved context]
            
            Vector --- Chroma
            Graph --- Neo4j
        end
        
        subgraph Agent_Logic ["Agentic Review"]
            Analyst[Hypothesis analyst<br/>Answers one hypothesis]
            Reviewer[Reviewer agent<br/>Validates answer quality]
            
            Analyst -->|verdict + evidence| Reviewer
            Reviewer -- "Retry if rejected (max 3 tries)" --> Analyst
        end
        
        Router -->|query embedding| Vector
        Router -->|graph query| Graph
        Vector --> Context
        Graph --> Context
        Context -->|context + contract + hypothesis| Analyst
    end

    %% Main Flow
    Start -->|contract text + prompt + history| ConvAgent
    ConvAgent -->|parsed query + session context| Router
    Reviewer -->|17 validated verdicts + evidence spans| Outputs[17 reviewed hypothesis outputs]
    
    Outputs --> Playbook[Playbook enrichment<br/>Adds policy context, risk flags, actions]
    Playbook --> Formatter[Runtrace formatter<br/>Structures all outputs per schema]
    Formatter --> Final([Final NDA review output])

    %% Styling
    style Start fill:#f9f9f9,stroke:#333
    style ConvAgent fill:#d1e9ff,stroke:#0056b3
    style Router fill:#d4edda,stroke:#28a745
    style Vector fill:#d4edda,stroke:#28a745
    style Graph fill:#d4edda,stroke:#28a745
    style Context fill:#f8f9fa,stroke:#333
    style Analyst fill:#e2d1f0,stroke:#6f42c1
    style Reviewer fill:#f8d7da,stroke:#dc3545
    style Playbook fill:#fff3cd,stroke:#856404
    style Formatter fill:#d4edda,stroke:#28a745
    style Final fill:#000,color:#fff
    style Loop fill:none,stroke:#333,stroke-dasharray: 5 5
```

### Key Changes:
- **Scope Expansion**: The "For each hypothesis (x17)" dotted boundary (represented by the `Loop` subgraph) now includes the **Retrieval Router**, **Vector RAG**, and **GraphRAG** pipelines.
- **Granular Retrieval**: This indicates that the system performs a targeted retrieval query for each specific hypothesis rather than a single bulk retrieval for the entire contract.
- **Agent Integration**: The Hypothesis Analyst and Reviewer Agent remain within the loop, receiving context tailored to each specific hypothesis.
