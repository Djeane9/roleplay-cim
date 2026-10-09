import streamlit as st 
from google import genai 
from google.genai import types
from groq import Groq 

st.set_page_config(page_title="Role-play CIM-ESUP", page_icon="icon.png")

client = genai.Client()
client_groq = Groq()

personnage = """ Tu es Mr. Ozias, recruteur d'une compagnie minière internationale. 
Tu fais passer un entretien d'embauche en anglais à un étudiant en ingénierie minière. 
Règles : 
- Parle uniquement en anglais, avec des phrases simples (niveau débutant à intermédiaire).
- Pose des questions sur les compétences techniques, l'expérience et la motivation du candidat.
- Pose une seule question à la fois, et réagis à la réponse de l'étudiant.
- Reste toujour dans ton personnage.
- Ne corrige pas ses erreurs : continue la conversation naturellement."""

# Nouveau : les consignes du bilan
evaluateur = """Tu es à la fois Mr. Ozias, recruteur minier, et un professeur d'anglais exigeant.
Tu reçois la transcription d'un entretien d'embauche en anglais avec un étudiant en ingénierie minière.
Réponds en français, dans ce format exact :
1. décision : candidat accepté ou candidat refusé, avec une phrase d'explication.
2. points forts : deux points positifs.
3. erreurs d'anglais : chaque erreur de l'étudiant, avec la correction et une explication.
4. suggestions : trois conseils concrets pour réussir un vrai entretien.
5. phrases utiles : un candidat qui répond en français, très peu ou hors sujet ne peut pas être accepté."""


premier_message="Good morning, and welcome. Please have a seat. Could you introduce yourself?"

modeles = ["gemini-flash-latest", "gemini-3.1-flash-lite","gemini-flash-lite-latest"]


def repondre(historique):
    contenu = [types.Content(role="user", parts=[types.Part(text="Start the interview.")])]
    for message in historique:
        if message["role"] == "user":
            role = "user"
        else:
            role = "model"
        contenu.append(types.Content(role=role, parts=[types.Part(text=message["content"])]))

    for modele in modeles:
        try:
            reponse = client.models.generate_content(
                model=modele,
                contents=contenu,
                config=types.GenerateContentConfig(system_instruction=personnage)
            )
            return reponse.text
        except Exception:
            pass

    messages = [{"role": "system", "content": personnage}] + historique 
    reponse_groq = client_groq.chat.completions.create(model="openai/gpt-oss-120b", messages=messages)
    return reponse_groq.choices[0].message.content

# Nouveau : la fonction qui produit bilan
def evaluer(transcription):
    for modele in modeles:
        try:
            reponse = client.models.generate_content(
                model=modele,
                contents=transcription,
                config=types.GenerateContentConfig(system_instruction=evaluateur)
            )
            return reponse.text
        except Exception:
            pass

    reponse_groq = client_groq.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": evaluateur},
            {"role": "user", "content": transcription},
        ],
    )
    return reponse_groq.choices[0].message.content


st.image("logo.png", width=300)
st.title("Role-play : entretien d'embauche")
st.write("Club de l'Ingénierie Minière (CIM-ESUP), section anglaise")
st.info(" Ta mission : convaincre Mr. Ozias de t'embaucher. Réponds en anglais à toutes ses questions, puis demande ton bilan après au moins 3 échanges.")
with st.sidebar:
    st.image("icone.png", width=80)
    st.markdown("### Conseils pour réussir")
    st.markdown("""
    - Fais des **phrases complètes**, pas un seul mot.
    - Commence chaque phrase par une **majuscule**, et écris **I** en majuscule.
    - Réponds à **chaque question** avant de demander ton bilan.
    - Reste **professionnel** : parle de tes compétences et de ta motivation.
    """)
    st.caption("Bridgineers - CIM-ESUP")

if "historique" not in st.session_state:
    st.session_state["historique"] = [{"role": "assistant", "content": premier_message}]

# nouveau : la mémoire du bilan 
if "bilan" not in st.session_state:
    st.session_state["bilan"] = None

# nouveau : les avatars 
for message in st.session_state["historique"]:
    if message["role"] == "assistant":
        avatar = "👔"
    else:
        avatar = "⛏️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# nouveau : tant que le bilan n'existe pas, l'entretien continue
if st.session_state["bilan"] is None:
    reponse_membre = st.chat_input("Écris ta réponse en anglais...")

    if reponse_membre:
        st.session_state["historique"].append({"role": "user", "content": reponse_membre})
        with st.chat_message("user", avatar="⛏️"):
            st.markdown(reponse_membre)
        with st.chat_message("assistant", avatar="👔"):
            with st.spinner("Mr. Ozias réfléchit..."):
                try:
                    texte = repondre(st.session_state["historique"])
                    st.markdown(texte)
                    st.session_state["historique"].append({"role": "assistant", "content": texte})
                except Exception as e:
                    st.error("Le personnage ne répond pas. Réessaie dans une minute.")
                    st.caption(f"Détail technique : {e}")

        # nouveau : le compteur d'échanges
    nb_echanges = 0
    for message in st.session_state["historique"]:
        if message["role"] == "user":
            nb_echanges = nb_echanges + 1
    st.caption(f"Échanges : {nb_echanges}")

    # nouveau : le bouton du bilan, à partir de 3 échanges
    if nb_echanges >= 3:
        if st.button("Terminer et voir mon bilan", type="primary"):
            lignes = []
            for message in st.session_state["historique"]:
                if message["role"] == "assistant":
                    lignes.append(f"Mr. Ozias : {message['content']}")
                else:
                    lignes.append(f"Étudiant : {message['content']}")
            transcription = "\n".join(lignes)
            with st.spinner("Mr. Ozias prépare ton bilan..."):
                try:
                    st.session_state["bilan"] = evaluer(transcription)
                    st.rerun()
                except Exception as e:
                    st.error("Le bilan n'a pas pu être généré. Réessaie dans une minute.")
                    st.caption(f"Détail technique : {e}")
else:
    # nouveau : l'affichage du bilan
    bilan = st.session_state["bilan"]
    if "refusé" in bilan.lower():
        st.error("Mr. Ozias ne t'a pas retenu cette fois. Lis ton bilan et retente ta chance !")
    elif "accepté" in bilan.lower():
        st.success("Félicitatiions, tu es embauché !")
        st.balloons()
    st.subheader("Ton bilan d'entretien")
    st.markdown(bilan)

if st.button("Recommencer l'entretien"):
    del st.session_state["historique"]
    del st.session_state["bilan"]
    st.rerun()