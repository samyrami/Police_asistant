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
        try:
            loader = DirectoryLoader(
                'data',
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()
        except Exception as e:
            st.error(f"Error loading PDFs: {str(e)}")
            documents = []
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



SYSTEM_PROMPT = """
Eres PoliciApp, un asistente especializado para oficiales de policía de tránsito en Colombia. 
Tu objetivo es proporcionar información legal precisa y contextualizada sobre infracciones de tránsito y comportamientos contrarios a la convivencia.

DIRECTRICES PARA INFORMES:

I. Clasificación de Infracciones:
A. Infracciones de Tránsito:
   - Monetarias (SMDLV o SMLMV)
   - No Monetarias (Suspensión, Retención)
   - Combinadas (Multa + Otra Acción)

B. Comportamientos Contrarios a la Convivencia:
   - Código Nacional de Seguridad y Convivencia Ciudadana
   - Medidas correctivas aplicables
   - Procedimientos específicos

II. Debido Proceso:
1. Identificación precisa de la infracción o comportamiento
2. Procedimiento de Imposición específico al caso
3. Derechos del ciudadano
4. Mecanismos de defensa aplicables

III. Formato de Respuesta:

• Tipo de Infracción: [Tránsito o Comportamiento Contrario]

• Descripción: [Descripción precisa del comportamiento]

• Base Legal: [Artículo específico (TEXTO COMPLETO)]

• Tipo de Sanción: [Monetaria/No Monetaria/Medida Correctiva]

• Cuantía: [Valor específico en SMDLV/SMLMV]

• Procedimiento: [Pasos específicos según el tipo de infracción]

• Medidas Inmediatas: [Acciones que debe tomar el agente]

• Derechos del Ciudadano: [Recursos y garantías específicas]

INSTRUCCIONES ESPECIALES:
1. SIEMPRE citar el artículo completo y textual de la norma
2. Diferenciar claramente entre infracciones de tránsito y comportamientos contrarios
3. Proporcionar el procedimiento específico según el tipo de infracción
4. Incluir las medidas inmediatas que debe tomar el agente
5. NO ASUMIR que todas las infracciones son de tránsito
6. Verificar el contexto antes de citar normas de tránsito
"""


def get_article_text(vector_store, article_reference):
    """
    Busca y retorna el texto completo de un artículo específico.
    """
    # Realizar una búsqueda específica por el número de artículo
    similar_docs = vector_store.similarity_search(
        f"Artículo {article_reference}",
        k=3
    )
    
    # Filtrar y extraer el texto completo del artículo
    for doc in similar_docs:
        content = doc.page_content
        if f"Artículo {article_reference}" in content:
            # Extraer el texto completo del artículo
            return content
    
    return None

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
    

def display_response(response_text, container):
    """Display the response using Streamlit components with article text."""
    # Simplemente mostrar el texto como markdown
    container.markdown(response_text)

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
    """Generate chat response using the selected LLM with improved context handling."""
    try:
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)
        
        # Primero, buscar contexto relevante en la base de datos
        relevant_context = search_laws(prompt)
        
        # Crear un prompt enriquecido con el contexto
        enhanced_prompt = f"""
        Consulta: {prompt}
        
        Contexto relevante encontrado en la base de datos:
        {relevant_context.to_string() if not relevant_context.empty else 'No se encontró contexto específico'}
        
        Por favor, proporciona una respuesta detallada basada en este contexto y las normas aplicables.
        """
        
        chat_model = ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            api_key=API_KEY,
            streaming=True,
            callbacks=[stream_handler]
        )
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=enhanced_prompt)
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


def ensure_directory_exists():
    """Ensure necessary directories exist."""
    directories = ["./faiss_index", "./leyes_pdf"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def check_startup_health():
    """Verify all components are ready."""
    try:
        # Check API key
        if not API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
            
        # Check directories
        ensure_directory_exists()
        
        # Check vector store
        vector_store = LawDocumentProcessor.load_vector_store()
        if vector_store is None:
            st.warning("Initializing knowledge base...")
            processor = LawDocumentProcessor()
            vector_store = processor.process_documents()
            
        return True
    except Exception as e:
        st.error(f"Startup health check failed: {str(e)}")
        return False

# Add this at the start of your main function
if not check_startup_health():
    st.stop()

def main():
    ensure_directory_exists()
    st.set_page_config(page_title="PoliciApp", layout="centered")
    st.write(logo, unsafe_allow_html=True)
    st.title("PoliciApp", anchor=False)
    st.markdown("**Asistente virtual para procedimientos de multas y comparendos**")
    
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