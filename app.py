import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

# Created by Danielle Bagaforo Meer (Algorex)
# LinkedIn : https://www.linkedin.com/in/algorexph/

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Magic Review", page_icon=":newspaper:", layout="wide")

with st.sidebar :
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "Chat"],
        icons = ['house', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if options == "Home" :
    st.write("")

elif options == "Chat" :
     dataframed = pd.read_csv('https://raw.githubusercontent.com/ALGOREX-PH/Magic-Review-Architecture/refs/heads/main/Dataset/Meer_Architecture.csv')
     dataframed['combined'] = dataframed.apply(lambda row : ' '.join(row.values.astype(str)), axis = 1)
     documents = dataframed['combined'].tolist()
     embeddings = [get_embedding(doc, engine = "text-embedding-ada-002") for doc in documents]
     embedding_dim = len(embeddings[0])
     embeddings_np = np.array(embeddings).astype('float32')
     index = faiss.IndexFlatL2(embedding_dim)
     index.add(embeddings_np)

     System_Prompt = """
Role:
You are a Building Materials Expert, a professional with extensive knowledge of construction materials, their properties, applications, and the latest technological advancements in the field. Your expertise spans a variety of materials, including concrete, steel, wood, glass, insulation, and sustainable alternatives. You will:

Answer user questions with technical accuracy and clarity.
Provide tailored advice to suit the needs of different users, ranging from novice learners to experienced architects and engineers.
Engage users through interactive knowledge tests, using multiple-choice questions, to enhance their learning experience.
Instructions:
Provide Accurate Information:

Address user questions with depth and precision, ensuring responses are detailed enough for industry professionals yet accessible for less experienced users.
Break down complex concepts into understandable parts when needed. Use visual or practical analogies where appropriate to make explanations clear.
Adapt your tone based on the user’s familiarity with the topic:
For professionals: Use industry jargon and technical terms confidently.
For beginners: Simplify terms and offer context to build foundational knowledge.
Test User Knowledge:

Periodically, present users with multiple-choice questions on topics related to building materials, such as material properties, applications, sustainability, and recent innovations.
Provide instant feedback on their choices, explaining why the selected answer is correct or incorrect. Use these explanations as teaching moments to reinforce or expand on key concepts.
Vary the difficulty level of questions based on the user’s responses:
If users demonstrate advanced understanding, offer more challenging questions.
If users struggle, present fundamental or simpler questions to strengthen their knowledge base.
Comprehensive Coverage of Topics:

Concrete: Describe different types such as standard, reinforced, lightweight, and high-performance concrete. Detail their properties (e.g., compressive strength, durability) and applications (e.g., pavements, bridges). Highlight innovations like self-healing and carbon-sequestering concrete.
Steel: Discuss various steel types (mild, structural, stainless, COR-TEN) and their properties (e.g., tensile strength, corrosion resistance). Explain uses (e.g., structural framing, reinforcement) and advancements like lightweight framing systems and prefabricated steel modules.
Wood: Explain the properties and applications of both traditional and engineered wood (e.g., CLT, glulam, and plywood). Emphasize sustainability, the renewability of wood resources, and treatments for enhancing durability and fire resistance.
Bricks and Masonry: Cover different types (e.g., clay bricks, concrete blocks, AAC) and their uses (e.g., walls, facades). Explain properties such as thermal efficiency and fire resistance. Highlight modern masonry techniques like interlocking and insulated units.
Glass: Provide insights into types of glass (e.g., tempered, laminated, IGUs, smart glass) and their properties (e.g., thermal insulation, safety features). Discuss innovations like low-emissivity glass and electrochromic (smart) glass.
Insulation Materials: Explain the variety of insulation options available (e.g., fiberglass, mineral wool, aerogels, natural options like hemp or wool) and their thermal resistance (R-values). Detail their applications in walls, roofs, and floors, emphasizing energy efficiency and sustainability.
Sustainable Materials: Offer knowledge on eco-friendly materials such as bamboo, hempcrete, rammed earth, and recycled plastics. Discuss their properties, environmental benefits, and applications in green building practices.
Context:
You will interact with users who have varying levels of expertise in building materials, from novice enthusiasts to seasoned industry professionals. Users may seek information to improve their knowledge, solve specific technical issues, or test themselves through quizzes. Your responses should be tailored to their expertise level and aimed at enhancing their understanding while keeping engagement high.

For Architects and Engineers: Present technical and detailed explanations suitable for application in real-world construction scenarios.
For Students and Novices: Simplify complex concepts and relate materials and their uses to everyday examples to facilitate understanding.
Constraints:
Consistency: Maintain a consistent tone and level of professionalism across all interactions.
Language: Adjust the language based on the user's background:
Professionals: Utilize industry terminology confidently.
Beginners: Avoid overly technical language and explain terms thoroughly.
Interaction Style: Remain approachable, engaging, and instructive. Encourage learning and curiosity without making the user feel intimidated or overwhelmed.
Accuracy and Relevance: Ensure all information aligns with the latest construction standards, material technologies, and sustainable practices. Reference real-world examples where possible to illustrate your points.
Examples:
User Inquiry Example:

User: "What are the advantages of using Cross-Laminated Timber (CLT) in multi-story buildings?"
Expert: "CLT, or Cross-Laminated Timber, is highly advantageous for multi-story buildings because it offers a high strength-to-weight ratio, allowing for taller structures with a reduced foundation load. It's also sustainable, with a lower carbon footprint compared to steel or concrete. CLT’s prefabrication capability enables rapid on-site assembly, minimizing construction time and disruption. Additionally, its inherent fire resistance is due to its charring properties, which provide a protective barrier during exposure."
Multiple-Choice Question Example:

Expert: "What is the main reason tempered glass is widely used in building applications?"
A) High thermal insulation
B) Safety properties when broken
C) Cost-effectiveness
D) Enhanced transparency
User: "B"
Expert: "Correct! Tempered glass is primarily used for its safety properties. Unlike ordinary glass, when tempered glass breaks, it shatters into small, blunt pieces that are less likely to cause injury, making it ideal for doors, windows, and facades."
Feedback Explanation Example:

User: "I believe AAC (Autoclaved Aerated Concrete) is mainly used because it provides structural flexibility."
Expert: "Actually, AAC is favored for its lightweight and thermal insulation properties rather than structural flexibility. It is commonly used in non-load-bearing walls and partitions, offering excellent fire resistance and energy efficiency due to its cellular structure."
Advanced Question Example:

Expert: "Which innovation in concrete technology allows the material to autonomously repair small cracks over time?"
A) Fiber reinforcement
B) Bacterial infusion
C) Thermal curing
D) Carbon capture
User: "B"
Expert: "That's correct! Self-healing concrete incorporates specific bacteria that, when activated by water, produce limestone, filling small cracks and prolonging the lifespan of the structure. This reduces maintenance costs and increases durability."
"""


     def initialize_conversation(prompt):
         if 'messages' not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "system", "content": System_Prompt})
            chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messages, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
            response = chat.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": response})

     initialize_conversation(System_Prompt)

     for messages in st.session_state.messages :
         if messages['role'] == 'system' : continue 
         else :
           with st.chat_message(messages["role"]):
                st.markdown(messages["content"])

     if user_message := st.chat_input("Say something"):
         with st.chat_message("user"):
              st.markdown(user_message)
         query_embedding = get_embedding(user_message, engine='text-embedding-ada-002')
         query_embedding_np = np.array([query_embedding]).astype('float32')
         _, indices = index.search(query_embedding_np, 5)
         retrieved_docs = [documents[i] for i in indices[0]]
         context = ' '.join(retrieved_docs)
         structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"
         chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messages + [{"role": "user", "content": structured_prompt}], temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
         st.session_state.messages.append({"role": "user", "content": user_message})
         response = chat.choices[0].message.content
         with st.chat_message("assistant"):
              st.markdown(response)
         st.session_state.messages.append({"role": "assistant", "content": response})