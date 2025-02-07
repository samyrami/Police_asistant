# PoliciApp - Asistente de Consulta Legal para Policías 👮‍♂️

## Descripción
PoliciApp es una aplicación de asistencia para oficiales de policía que facilita la consulta rápida y precisa de leyes, normas y procedimientos. Utiliza inteligencia artificial para procesar consultas en lenguaje natural y proporcionar respuestas basadas en la legislación colombiana vigente.

## Características Principales 🚔
- Consulta rápida de infracciones y multas
- Base de conocimientos integrada con leyes colombianas
- Búsqueda inteligente en documentos legales
- Interfaz conversacional intuitiva
- Generación de respuestas estructuradas para multas
- Sistema de búsqueda en tiempo real

## Requisitos Previos 📋
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Una clave API válida de OpenAI
- PDFs con la legislación relevante

## Instalación 🛠️

1. **Clonar el repositorio**
```bash
git clone https://github.com/samyrami/Police_asistant?tab=readme-ov-file
cd policiapp
```

2. **Crear y activar un entorno virtual**
```bash
python -m venv venv

# En Windows
venv\Scripts\activate

# En macOS/Linux
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar las variables de entorno**
- Crear un archivo `.env` en el directorio raíz
- Agregar tu clave API de OpenAI:
```
OPENAI_API_KEY=tu-clave-api-aqui
```

5. **Preparar los documentos**
- Crear un directorio `leyes_pdf`
- Colocar los PDFs de leyes en el directorio

## Estructura del Proyecto 📁
```
policiapp/
├── app.py                 # Aplicación principal
├── .env                   # Variables de entorno
├── config.toml           # Configuración de Streamlit
├── requirements.txt      # Dependencias
├── html_template.py      # Plantillas HTML
├── README.md            # Documentación
└── leyes_pdf/           # Directorio de documentos legales
    ├── codigo_transito.pdf
    ├── codigo_policia.pdf
    └── otras_leyes.pdf
```

## Uso 🚀

1. **Iniciar la aplicación**
```bash
streamlit run app.py
```

2. **Tipos de consultas**

- **Consulta de multas**:
```
MULTA: Conductor en estado de embriaguez
```

- **Consulta general**:
```
¿Cuál es el procedimiento para una inspección vehicular?
```

3. **Búsqueda en base de datos**
- Utilizar la barra de búsqueda lateral para consultas rápidas

## Funcionalidades Principales 💡

### Procesamiento de Multas
- Identificación automática de infracciones
- Cálculo de sanciones
- Procedimientos a seguir
- Derechos del infractor

### Base de Conocimientos
- Indexación automática de documentos
- Búsqueda semántica
- Citación de fuentes
- Actualizaciones periódicas

## Mantenimiento 🔧

### Actualización de la Base de Conocimientos
1. Agregar nuevos PDFs al directorio `leyes_pdf`
2. Reiniciar la aplicación
3. El sistema procesará automáticamente los nuevos documentos

### Respaldo de Datos
- Los índices vectoriales se almacenan en `faiss_index`
- Hacer copias de seguridad periódicas de este directorio

## Solución de Problemas ⚠️

### Errores Comunes
1. **Error de API Key**:
   - Verificar el archivo `.env`
   - Confirmar que la clave API es válida

2. **Error de Procesamiento de PDFs**:
   - Verificar que los PDFs no estén corruptos
   - Confirmar permisos de lectura

3. **Error de Memoria**:
   - Reducir el tamaño de los chunks de texto
   - Optimizar la cantidad de documentos

## Contribución 🤝
Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Crea un Pull Request

## Licencia 📄
Este proyecto está bajo la Licencia MIT - ver el archivo `LICENSE.md` para detalles

## Contacto 📧
Para soporte o consultas:
- Email: serr73094@gmail.com

## Reconocimientos 🎉
- OpenAI por la API de GPT
- Streamlit por el framework
- Langchain por las herramientas de procesamiento
- FAISS por el sistema de búsqueda vectorial