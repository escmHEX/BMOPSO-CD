# 4.1.2. Software y Librerías

El sistema fue implementado sobre un entorno Linux, utilizando las siguientes herramientas y bibliotecas:

Sistema Operativo: Ubuntu 20.04 LTS (64 bits).

Lenguaje de Programación: Python 3.13.13.

Gestor de Modelos: Ollama 0.30.10.

Modelos de Lenguaje: Llama 3.1 8B (llama3.1:8b), Gemma (gemma4:e4b),
Qwen 3.5 2B (qwen3.5:2b) y Qwen 3.5 4B (qwen3.5:4b).

Librerías Principales:

- Generación y Orquestación: Ollama 0.6.2 para la comunicación con los
  modelos locales, y Asyncio (librería estándar) para la ejecución asíncrona
  y concurrente.
- Configuración y Ejecución: PyYAML 6.0.3 para la gestión de configuraciones,
  junto con librerías estándar de Python para serialización, línea de comandos,
  manejo de rutas y persistencia de resultados.
- Representación Semántica y Cálculo Numérico: Sentence-Transformers 5.5.1
  con all-MiniLM-L6-v2 para embeddings semánticos, y NumPy 2.4.6 para
  operaciones vectoriales y cálculo de métricas internas.
- Procesamiento de Lenguaje Natural: Transformers 5.9.0 y Torch 2.12.0 para
  modelos auxiliares basados en Transformers; spaCy 3.8.14 con en_core_web_sm
  3.8.0 para análisis lingüístico; y NLTK 3.9.4 con WordNet para recursos
  léxicos.
- Recursos Léxicos y Almacenamiento Local: PPDB 2.0 S-all como recurso de
  paráfrasis, indexado mediante SQLite para búsquedas locales eficientes.
- Métricas y Resultados: Pandas 3.0.3 para la exportación de resultados
  tabulares y scikit-learn 1.8.0 para métricas auxiliares basadas en clustering.
