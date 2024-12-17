import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from html_template import logo
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import re

# Load environment variables
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    st.error("Error: OPENAI_API_KEY not found in environment variables")
    st.stop()
    
class LawDocumentProcessor:
    def __init__(self, pdf_directory="./leyes_pdf"):
        self.pdf_directory = pdf_directory
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
    def load_documents(self):
        """Carga todos los PDFs del directorio especificado"""
        loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        return documents
    
    def process_documents(self):
        """Procesa los documentos y crea el almacén de vectores"""
        documents = self.load_documents()
        texts = self.text_splitter.split_documents(documents)
        
        # Crear el almacén de vectores con FAISS
        vector_store = FAISS.from_documents(texts, self.embeddings)
        
        # Guardar el índice localmente
        vector_store.save_local("faiss_index")
        return vector_store
    
    @staticmethod
    def load_vector_store():
        """Carga el almacén de vectores guardado"""
        try:
            if os.path.exists("faiss_index"):
                vector_store = FAISS.load_local(
                    "faiss_index", 
                    OpenAIEmbeddings(), 
                    allow_dangerous_deserialization=True  # Add this parameter
                )
                return vector_store
            return None
        except Exception as e:
            st.error(f"Error cargando el vector store: {str(e)}")
            return None

def setup_retrieval_chain(vector_store):
    """Configura la cadena de recuperación para consultas"""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return retrieval_chain

# ... (código anterior del app.py) ...

# Añadir después de la inicialización de variables globales
try:
    vector_store = LawDocumentProcessor.load_vector_store()
    if vector_store is None:
        st.warning("Procesando documentos legales por primera vez...")
        processor = LawDocumentProcessor()
        vector_store = processor.process_documents()
    
    retrieval_chain = setup_retrieval_chain(vector_store)
except Exception as e:
    st.error(f"Error cargando la base de conocimientos: {str(e)}")
    vector_store = None
    retrieval_chain = None

def get_chat_response(prompt, temperature=0.3):
    """Generate chat response using the selected LLM and law knowledge base."""
    try:
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)
        
        # Si es una consulta de multa, usar el formato específico
        if prompt.upper().startswith("MULTA:"):
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            chat_model = ChatOpenAI(
                model="gpt-4",
                temperature=temperature,
                api_key=API_KEY,
                streaming=True,
                callbacks=[stream_handler]
            )
            
            response = chat_model.invoke(messages)
        else:
            # Para consultas generales, usar la base de conocimientos
            result = retrieval_chain({"question": prompt})
            response = result["answer"]
            
            # Agregar las fuentes de la información
            if "source_documents" in result:
                response += "\n\nFuentes consultadas:\n"
                for doc in result["source_documents"]:
                    response += f"- {doc.metadata.get('source', 'Documento legal')}\n"
        
        return stream_handler.text if prompt.upper().startswith("MULTA:") else response
        
    except Exception as e:
        st.error(f"Error generando respuesta: {str(e)}")
        return "Lo siento, ocurrió un error al procesar su solicitud."

SYSTEM_PROMPT = """
Eres PoliciApp, un asistente especializado para oficiales de policía de tránsito en Colombia. 
Tu objetivo es proporcionar información legal detallada sobre infracciones de tránsito.

DIRECTRICES PARA INFORMES DE INFRACCIONES:

I. Clasificación de Sanciones:
- Monetarias (SMDLV o SMLMV)
- No Monetarias (Suspensión, Retención)
- Combinadas (Multa + Otra Acción)

II. Debido Proceso:
1. Identificación de la Infracción
2. Procedimiento de Imposición
3. Derechos del Infractor
4. Mecanismos de Defensa

III. Formato de Respuesta Detallado:

| Campo | Descripción Detallada |
|-------|----------------------|
| Infracción | Descripción precisa |
| Base Legal | Artículo específico |
| Tipo de Sanción | Monetaria/No Monetaria/Combinada |
| Cuantía | Valor en SMDLV/SMLMV o descripción |
| Procedimiento | Pasos legales a seguir |
| Consecuencias Adicionales | Suspensión, retención, etc. |
| Derechos del Infractor | Recursos legales |

INSTRUCCIONES ESPECIALES:
- Citar normas exactas
- Describir procedimiento detallado
- Explicar consecuencias legales
- Indicar mecanismos de defensa
"""

def calcular_sancion(detalles_sancion):
    """
    Calcula y detalla sanciones complejas
    
    :param detalles_sancion: Diccionario con detalles de la sanción
    :return: Diccionario con información detallada
    """
    # Valor del SMMLV para 2024 (actualizable)
    SMMLV_2024 = 1_300_000
    SMDLV_2024 = SMMLV_2024 / 30

    resultado = {
        "sanciones": [],
        "procedimiento_detallado": "",
        "derechos_infractor": ""
    }

    # Manejo de sanciones monetarias
    if "monetaria" in detalles_sancion:
        unidad = detalles_sancion["monetaria"].get("unidad", "SMDLV")
        cantidad = detalles_sancion["monetaria"].get("cantidad", 0)
        
        if unidad == "SMDLV":
            valor_total = SMDLV_2024 * cantidad
            resultado["sanciones"].append({
                "tipo": "Multa",
                "unidad": "SMDLV",
                "cantidad": cantidad,
                "valor": f"${valor_total:,.0f}"
            })

    # Sanciones no monetarias
    if "no_monetaria" in detalles_sancion:
        sanciones_no_monetarias = detalles_sancion["no_monetaria"]
        for sancion in sanciones_no_monetarias:
            resultado["sanciones"].append({
                "tipo": sancion,
                "descripcion": {
                    "Suspensión de Licencia": "Retiro temporal del derecho a conducir",
                    "Retención de Vehículo": "Inmovilización del vehículo en depósito oficial"
                }.get(sancion, sancion)
            })

    # Procedimiento detallado
    resultado["procedimiento_detallado"] = """
    Procedimiento Estándar:
    1. Detección de la Infracción
    2. Registro en Formato Único de Comparendo
    3. Notificación al Infractor
    4. Opciones de Pago/Descargos
    5. Posible Impugnación
    """

    # Derechos del Infractor
    resultado["derechos_infractor"] = """
    Derechos del Infractor:
    - Derecho a presentar descargos
    - Solicitar práctica de pruebas
    - Interponer recursos de reposición
    - Acceso a defensa técnica
    - Principio de presunción de inocencia
    """

    return resultado
    
    
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_response(self.text, self.container)

def extract_table_data(markdown_text):
    """Extract table data from markdown and convert to DataFrame."""
    try:
        # Find table in text using regex
        table_pattern = r'\|.*\|'
        table_rows = re.findall(table_pattern, markdown_text)
        
        if not table_rows:
            return None, None
            
        # Process table rows
        headers = ['Campo', 'Valor']
        data = []
        
        for row in table_rows[2:]:
            values = [col.strip() for col in row.split('|')[1:-1]]
            if len(values) == 2:
                data.append(values)
        
        df = pd.DataFrame(data, columns=headers)
        
        pre_table = markdown_text.split('|')[0].strip()
        post_table = markdown_text.split('|')[-1].strip()
        other_text = f"{pre_table}\n\n{post_table}".strip()
        
        return df, other_text
    except Exception as e:
        st.error(f"Error procesando la tabla: {str(e)}")
        return None, None

def display_response(response_text, container):
    """Display the response using Streamlit components."""
    if '|' in response_text:
        df, other_text = extract_table_data(response_text)
        if df is not None:
            if other_text:
                container.markdown(other_text)
            
            container.markdown("### Detalles de la Infracción")
            
            styled_df = df.style.set_properties(**{
                'background-color': '#f0f2f6',
                'color': '#1f1f1f',
                'border': '2px solid #1e3d59'
            })
            
            container.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Campo": st.column_config.TextColumn(
                        "Campo",
                        help="Categoría de la información",
                        width="medium",
                    ),
                    "Valor": st.column_config.TextColumn(
                        "Valor",
                        help="Información detallada",
                        width="large",
                    )
                }
            )
        else:
            container.markdown(response_text)
    else:
        container.markdown(response_text)

def search_laws(query):
    """Buscar en los documentos legales vectorizados con más detalles."""
    if vector_store is None:
        st.error("Base de conocimientos no inicializada")
        return None
    
    # Realizar una búsqueda de similitud en el vector store
    similar_docs = vector_store.similarity_search(query, k=5)
    
    # Preparar los resultados
    results = []
    for doc in similar_docs:
        # Extraer información del contexto
        content = doc.page_content
        source = doc.metadata.get('source', 'Documento desconocido')
        page = doc.metadata.get('page', 'N/A')
        
        results.append({
            'Artículo/Código': f"Fuente: {source}, Página: {page}",
            'Contenido Relevante': content[:500] + '...',  # Mostrar un extracto más largo
        })
    
    # Convertir a DataFrame para mostrar resultados
    results_df = pd.DataFrame(results)
    return results_df


def get_chat_response(prompt, temperature=0.3):
    """Generar respuesta usando la base de conocimientos legales."""
    try:
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)
        
        # Configurar el modelo de chat
        chat_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=API_KEY,
            streaming=True,
            callbacks=[stream_handler]
        )
        
        # Preparar mensajes
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        # Procesar consulta de multa
        if prompt.upper().startswith("MULTA:"):
            multa_content = prompt[6:].strip()
            
            # Invocar modelo
            response_raw = chat_model.invoke(messages).content
            
            # Ejemplo de estructura para análisis de sanciones
            detalles_sancion = {
                "monetaria": {"unidad": "SMDLV", "cantidad": 15},
                "no_monetaria": ["Suspensión de Licencia", "Retención de Vehículo"]
            }
            
            # Calcular sanción
            calculo_sancion = calcular_sancion(detalles_sancion)
            
            # Formatear respuesta completa
            respuesta_completa = f"""
{response_raw}

### Detalles de la Sanción:

**Sanciones Aplicadas:**
{chr(10).join([f"- {s.get('tipo')}: {s.get('descripcion', s.get('valor', 'N/A'))}" for s in calculo_sancion['sanciones']])}

**Procedimiento Detallado:**
{calculo_sancion['procedimiento_detallado']}

**Derechos del Infractor:**
{calculo_sancion['derechos_infractor']}
            """
            
            return respuesta_completa
        
        # Resto del código de recuperación de información permanece igual
        
    except Exception as e:
        st.error(f"Error generando respuesta: {str(e)}")
        return "Lo siento, ocurrió un error al procesar su solicitud."

def get_chat_response(prompt, temperature=0.3):
    """Generate chat response using the selected LLM."""
    try:
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)
        
        chat_model = ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            api_key=API_KEY,
            streaming=True,
            callbacks=[stream_handler]
        )
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        if "messages" in st.session_state:
            for msg in st.session_state.messages[-3:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(SystemMessage(content=msg["content"]))
        
        response = chat_model.invoke(messages)
        return stream_handler.text
        
    except Exception as e:
        st.error(f"Error generando respuesta: {str(e)}")
        return "Lo siento, ocurrió un error al procesar su solicitud."

def main():
    st.set_page_config(page_title="PoliciApp", layout="centered")
    st.write(logo, unsafe_allow_html=True)
    st.title("PoliciApp - Asistente de Tránsito", anchor=False)
    st.markdown("**Asistente virtual para consulta rápida de infracciones y procedimientos de tránsito**")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.markdown("""
        **Bienvenido al Sistema de Consulta de Infracciones**
        
        Esta herramienta está diseñada para ayudarte a:
        - Consultar rápidamente infracciones de tránsito
        - Verificar sanciones aplicables
        - Conocer el procedimiento correcto
        - Informar sobre derechos del infractor
        
        **Para comenzar:**
        1. Escribe 'MULTA:' seguido de la situación
        2. O busca directamente en la base de datos de infracciones
        """)
        
        # Búsqueda en base de datos
        search_query = st.text_input("Buscar en base de datos de infracciones:")
        if search_query:
            results = search_laws(search_query)
            if not results.empty:
                st.dataframe(results)
            else:
                st.info("No se encontraron resultados")

        if st.button("Borrar Historial"):
            st.session_state.messages = []
            st.experimental_rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and '|' in message["content"]:
                display_response(message["content"], st)
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Describe la situación... (Inicia con MULTA: para procesar una infracción)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👮"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🚓"):
            is_multa = prompt.upper().startswith("MULTA:")
            if is_multa:
                multa_content = prompt[6:].strip()
                response = get_chat_response(multa_content)
            else:
                response = get_chat_response(prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()