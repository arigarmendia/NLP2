# NLP2 (CEIA-LLMIAG)

## Programa de la materia 

1. Repaso de Transformers, Arquitectura y Tokenizers.
2. Arquitecturas de LLMs, Transformer Decoder.
3. Ecosistema actual, APIs, costos, HuggingFace y OpenAI. Evaluación de LLMs.
4. MoEs, técnicas de prompts.
5. Modelos locales y uso de APIs.
6. RAG, vector DBs, chatbots y práctica.
7. Agentes, fine-tuning y práctica. 
8. LLMs de Razonamiento. Optimización, Generación multimodal y práctica.

---
### Trabajo práctico Nro 1:

[TP1 - TinyGPT con Mixture of Experts (MoE)](/TP1/TP1-TinyGPT.ipynb) 

---
### Objetivo
Implementar desde cero un modelo tipo GPT (decoder-only transformer) y extenderlo a una arquitectura Mixture of Experts (MoE). Se incluyen dos tareas:

### Task I - Generación de texto avanzada y pruebas con:

- Entrenamiento con 2 y 10 épocas.
- Pruebas con 
    - Greedy decoding
    - Temperature sampling
    - Top-k sampling
    - Top-p sampling
    - Combinaciones de las anteriores

### Task II - Arquitectura Mixture of Experts

- Conversión del TinyGPT original a un TinyGPT-MoE con 4 heads.
- Análisis comparativo de mapas de atención:

    a. [Resultados con 2 épocas](#visualizing-attention---2-epochs)
    
    b. [Resultados con 5 épocas](#visualizing-attention---5-epochs)

### Características principales:
- Modelo pequeño (~10M parámetros)
- Tokenizer character-level 100% manual (vocabulario de 65 tokens)
- Dataset: TinyShakespeare (100k caracteres)
- Entrenamiento en Apple Silicon (MPS) con bfloat16 + AMP



