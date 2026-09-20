# Tarjeta del modelo usado por el proyecto

## Identidad y arquitectura

El modelo configurado es `stabilityai/sd-turbo`, revisión `b261bac6fd2cf515557d5d0707481eafa0485ec2`. Es una destilación de Stable Diffusion 2.1 mediante Adversarial Diffusion Distillation. Se usa para texto a imagen a 512 x 512, normalmente en un paso y con CFG 0. [Tarjeta del autor](https://huggingface.co/stabilityai/sd-turbo).

Las configuraciones descargadas declaran un text encoder de dimensión 1024 y 77 posiciones, UNet con cross-attention de dimensión 1024, cuatro canales latentes y scheduler Euler con espaciado `trailing`. La implementación local ejecuta FP32 en CPU. Los detalles del flujo están en [README.md](README.md#arquitectura-técnica).

El decoder predeterminado es `madebyollin/taesd`, revisión `614f76814bbe30edbe2e627ace1c2234c81a2c0e`. TAESD es una aproximación compacta al autoencoder de Stable Diffusion; puede modificar detalle y color respecto al VAE original. No es el autoencoder de SDXL. [Recurso TAESD](https://huggingface.co/madebyollin/taesd).

## Entradas y salidas

- Entrada: prompt de texto; el tokenizador real limita la secuencia a 77 posiciones.
- Aleatoriedad: semilla explícita o generada localmente y registrada en el PNG.
- Salida: imagen RGB, guardada como PNG con metadatos de procedencia y parámetros.
- Resolución inicial: 512 x 512; reducirla puede empeorar la composición.
- CFG: 0 para SD-Turbo. El negative prompt no afecta a esa configuración.

## Limitaciones y evaluación

Un paso reduce el número de evaluaciones de la UNet, no garantiza una latencia concreta. La velocidad y la memoria dependen de la CPU, RAM disponible, resolución, VAE, verificador y bibliotecas. El proyecto no presenta tablas de rendimiento como mediciones sin un resultado de benchmark que las respalde.

Pueden aparecer errores de anatomía, texto ilegible, objetos incoherentes, variaciones de estilo y sesgos heredados del entrenamiento. Las salidas no prueban hechos ni reproducen necesariamente el prompt con exactitud. No se han realizado aquí evaluaciones demográficas ni una validación sistemática de calidad. Las limitaciones descritas por el autor siguen siendo aplicables. [Limitaciones del modelo](https://huggingface.co/stabilityai/sd-turbo#limitations).

## Perfiles y controles locales

`sfw` utiliza el verificador visual de `CompVis/stable-diffusion-safety-checker`, revisión `cb41f3a270d63d454d385fc2e4f571c487c253c5`, preparado dentro de `models/`. Una salida marcada provoca un error explícito y no se guarda. `nsfw` no carga ese clasificador. Ambos perfiles mantienen los patrones de `src/guardrails.py`.

Los patrones y el clasificador son controles heurísticos con falsos positivos y negativos, no garantías de detección de edad, identidad o consentimiento. La aplicación permanece en loopback y no crea un servicio público.

## Distribución y licencias

Los pesos no se incluyen en el repositorio. La preparación descarga los archivos de licencia disponibles junto a los recursos. No se debe asumir que SD-Turbo tiene la licencia de SD 1.5. Consulta la [licencia incluida en la revisión de SD-Turbo](https://huggingface.co/stabilityai/sd-turbo/blob/b261bac6fd2cf515557d5d0707481eafa0485ec2/LICENSE.md) y las condiciones publicadas por cada autor antes de redistribuir o utilizar comercialmente los pesos.

La instalación, configuración offline, pruebas y errores conocidos están documentados en [README.md](README.md). Los resultados verificables de esta revisión se registran en [REVIEW.md](REVIEW.md).
