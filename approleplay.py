import streamlit as st
from google import genai
from google.genai import types
from groq import Groq
from gtts import gTTS
import edge_tts
import asyncio
import base64
import html
import io
import json
import time

st.set_page_config(page_title="Role-play CIM-ESUP", page_icon="icone.png", layout="centered")

client = genai.Client()
client_groq = Groq()

# ---------- Réglages ----------
MODELES = ["gemini-flash-latest", "gemini-3.1-flash-lite", "gemini-flash-lite-latest"]
NB_QUESTIONS = 5        # nombre de questions de l'entretien
MIN_POUR_BILAN = 3      # nombre de réponses avant de pouvoir demander le bilan

# Couleurs Bridgineers
BLEU = "#1652C9"
MARINE = "#0A2A6B"
ROUGE = "#D7192F"
GRIS = "#5B6B8C"
BORDURE = "#D9DEEA"

PREMIER_MESSAGE = "Good morning, and welcome. Please have a seat. Could you introduce yourself?"

NIVEAUX = {
    "Débutant": "Parle avec des phrases très courtes et très simples (niveau A2), avec des mots courants. Sois encourageant.",
    "Intermédiaire": "Parle avec des phrases simples mais naturelles (niveau B1-B2).",
}


# ---------- Consignes pour l'IA ----------
def consignes_personnage(niveau):
    return f"""Tu es Mr. Ozias, recruteur d'une compagnie minière internationale.
Tu fais passer un entretien d'embauche en anglais à un étudiant en ingénierie minière.
Règles :
- Parle uniquement en anglais. {NIVEAUX[niveau]}
- Pose des questions sur les compétences techniques, l'expérience et la motivation du candidat.
- Pose une seule question à la fois. Réagis à la réponse de l'étudiant en une phrase maximum, puis pose la question suivante.
- L'entretien compte {NB_QUESTIONS} questions au total, présentation comprise.
- Après la {NB_QUESTIONS}e réponse de l'étudiant, remercie-le et conclus l'entretien, sans poser de nouvelle question.
- Reste toujours dans ton personnage.
- Ne corrige pas ses erreurs : continue la conversation naturellement."""


def consignes_evaluateur(niveau):
    if niveau == "Débutant":
        severite = ("L'étudiant est débutant : sois bienveillant. Accepte-le s'il répond en anglais "
                    "à toutes les questions de façon compréhensible, même avec des fautes.")
    else:
        severite = "L'étudiant est de niveau intermédiaire : sois exigeant mais juste."
    return f"""Tu es à la fois Mr. Ozias, recruteur minier, et un professeur d'anglais.
Tu reçois la transcription d'un entretien d'embauche en anglais avec un étudiant en ingénierie minière.
Une partie des réponses a été dictée à voix haute : ignore la ponctuation et les majuscules.
{severite}
Un candidat qui répond en français, très peu ou hors sujet ne peut pas être accepté.
Tutoie l'étudiant. Écris en français, sauf les phrases en anglais.
Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte autour, avec exactement ces clés :
{{
  "decision": "accepte" ou "refuse",
  "explication": "une phrase qui explique la décision",
  "notes": {{"anglais": entier de 0 à 10, "contenu": entier de 0 à 10, "assurance": entier de 0 à 10}},
  "points_forts": ["deux phrases courtes"],
  "corrections": [{{"dit": "phrase exacte de l'étudiant", "mieux": "phrase corrigée", "pourquoi": "explication courte"}}],
  "style": [{{"dit": "...", "mieux": "...", "pourquoi": "..."}}],
  "conseils": ["trois conseils concrets pour réussir un vrai entretien"],
  "phrases_utiles": ["trois phrases en anglais utiles en entretien minier"]
}}
"corrections" contient UNIQUEMENT les vraies fautes (grammaire, conjugaison, vocabulaire).
"style" contient les phrases correctes qui pourraient sonner plus naturelles ou plus professionnelles.
"assurance" mesure si l'étudiant développe ses réponses avec des phrases complètes et des exemples."""


CONSIGNES_AIDE = """Tu aides un étudiant francophone qui apprend l'anglais, pendant un entretien d'embauche.
Tu reçois la dernière phrase du recruteur.
Réponds UNIQUEMENT avec un objet JSON valide :
{"traduction": "traduction française de la phrase du recruteur",
 "debut": "un début de réponse en anglais, de 5 à 10 mots, qui se termine par ..."}"""


# ---------- Fonctions IA ----------
def lire_json(texte):
    debut = texte.find("{")
    fin = texte.rfind("}")
    return json.loads(texte[debut:fin + 1])


def demander_ia(consignes, texte, en_json=False):
    """Une question simple à l'IA : Gemini d'abord, Groq en secours."""
    reglages = {"system_instruction": consignes}
    if en_json:
        reglages["response_mime_type"] = "application/json"
    for modele in MODELES:
        try:
            reponse = client.models.generate_content(
                model=modele, contents=texte,
                config=types.GenerateContentConfig(**reglages),
            )
            return reponse.text
        except Exception:
            pass
    reponse_groq = client_groq.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "system", "content": consignes}, {"role": "user", "content": texte}],
    )
    return reponse_groq.choices[0].message.content


def repondre(historique, niveau):
    """La réplique suivante de Mr. Ozias."""
    consignes = consignes_personnage(niveau)
    contenu = [types.Content(role="user", parts=[types.Part(text="Start the interview.")])]
    for message in historique:
        role = "user" if message["role"] == "user" else "model"
        contenu.append(types.Content(role=role, parts=[types.Part(text=message["content"])]))
    for modele in MODELES:
        try:
            reponse = client.models.generate_content(
                model=modele, contents=contenu,
                config=types.GenerateContentConfig(system_instruction=consignes),
            )
            return reponse.text
        except Exception:
            pass
    messages = [{"role": "system", "content": consignes}] + historique
    reponse_groq = client_groq.chat.completions.create(model="openai/gpt-oss-120b", messages=messages)
    return reponse_groq.choices[0].message.content


@st.cache_data(show_spinner=False)
def aider(question):
    try:
        return lire_json(demander_ia(CONSIGNES_AIDE, question, en_json=True))
    except Exception:
        return None


def evaluer(transcription, niveau):
    texte = demander_ia(consignes_evaluateur(niveau), transcription, en_json=True)
    try:
        return lire_json(texte)
    except Exception:
        return {"brut": texte}


def transcrire(audio):
    resultat = client_groq.audio.transcriptions.create(
        file=("reponse.wav", audio.getvalue()),
        model="whisper-large-v3-turbo",
        language="en",
        prompt="Job interview for a mining engineering student.",
    )
    return resultat.text.strip()


@st.cache_data(show_spinner=False)
def parler(texte, lent):
    try:
        async def generer():
            audio = b""
            voix = edge_tts.Communicate(texte, "en-GB-RyanNeural", rate="-15%" if lent else "+0%")
            async for morceau in voix.stream():
                if morceau["type"] == "audio":
                    audio = audio + morceau["data"]
            return audio
        return asyncio.run(generer())
    except Exception:
        fichier = io.BytesIO()
        gTTS(text=texte, lang="en", slow=lent).write_to_fp(fichier)
        return fichier.getvalue()


# ---------- Mémoire de l'appli ----------
def nouvel_entretien(niveau):
    st.session_state["etape"] = "entretien"
    st.session_state["niveau"] = niveau
    st.session_state["historique"] = [{"role": "assistant", "content": PREMIER_MESSAGE}]
    st.session_state["en_attente"] = None
    st.session_state["bilan"] = None
    st.session_state["debut"] = time.time()
    st.session_state["a_jouer"] = True
    st.session_state["ballons"] = False
    st.session_state["erreur"] = None


def retour_accueil():
    st.session_state["etape"] = "accueil"


if "etape" not in st.session_state:
    st.session_state["etape"] = "accueil"
    st.session_state["num_micro"] = 0


# ---------- Design ----------
@st.cache_resource
def charger_logo():
    try:
        with open("icone.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def afficher_html(code):
    lignes = [ligne.strip() for ligne in code.splitlines() if ligne.strip()]
    st.markdown("\n".join(lignes), unsafe_allow_html=True)


afficher_html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Montserrat:wght@700&display=swap');
html, body, .stMarkdown, button, input, textarea {{font-family: 'Inter', sans-serif;}}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], footer {{display: none;}}
[data-testid="stHeader"] {{display: none;}}
.block-container, [data-testid="stMainBlockContainer"] {{padding-top: 2rem !important; padding-bottom: 3rem; max-width: 560px;}}
.entete {{display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;}}
.marque {{display: flex; align-items: center; gap: 10px;}}
.marque img {{width: 46px; height: 46px; flex-shrink: 0; object-fit: contain; border-radius: 10px; background: #fff; border: 1px solid {BORDURE};}}
.t1 {{margin: 0; font-family: 'Montserrat', sans-serif; font-weight: 700; font-size: 16px; color: {MARINE}; line-height: 1.2;}}
.t2 {{margin: 0; font-size: 12px; color: {GRIS}; line-height: 1.3;}}
.pastille {{font-size: 12px; background: #E3EBFA; color: {MARINE}; padding: 4px 10px; border-radius: 999px; white-space: nowrap;}}
.carte {{background: #fff; border: 1px solid {BORDURE}; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; color: #1B2440;}}
.carte h4 {{margin: 0 0 8px; font-size: 15px; color: {MARINE}; padding: 0;}}
.carte ul {{margin: 0; padding-left: 18px;}}
.carte li {{margin-bottom: 4px;}}
.recruteur {{display: flex; gap: 12px; align-items: flex-start;}}
.avatar {{width: 44px; height: 44px; flex-shrink: 0; border-radius: 50%; background: {MARINE}; color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 600;}}
.nom {{margin: 0; font-weight: 600; color: {MARINE}; font-size: 14px;}}
.nom span {{font-weight: 400; color: {GRIS};}}
.question {{margin: 4px 0 0; font-size: 17px; line-height: 1.5; color: #1B2440;}}
.petit {{margin: 2px 0 0; font-size: 13px; color: {GRIS};}}
.etat {{text-align: center; font-size: 15px; font-weight: 600; color: {MARINE}; margin: 10px 0 2px;}}
.sous-etat {{text-align: center; font-size: 13px; color: {GRIS}; margin: 0 0 10px;}}
.st-key-zone_micro [data-testid="stAudioInput"] {{border: 2px solid #C7D6F5; border-radius: 16px; background: #fff;}}
.stButton button, .stFormSubmitButton button {{width: 100%; min-height: 48px; border-radius: 10px; font-weight: 600;}}
.st-key-btn_bilan button {{background: {ROUGE} !important; border-color: {ROUGE} !important; color: #fff !important;}}
.note {{display: flex; justify-content: space-between; font-size: 14px; color: {MARINE};}}
.barre {{height: 8px; border-radius: 4px; background: #E3EBFA; overflow: hidden; margin: 4px 0 12px;}}
.barre div {{height: 100%; background: {BLEU}; border-radius: 4px;}}
.correction {{border-top: 1px solid {BORDURE}; padding-top: 8px; margin-top: 8px; font-size: 14px;}}
.correction:first-of-type {{border-top: none; padding-top: 0; margin-top: 0;}}
.dit {{color: {ROUGE}; text-decoration: line-through;}}
.mieux {{color: {BLEU}; font-weight: 600;}}
.verdict-ok {{background: #E3EBFA; border-color: {BLEU};}}
.verdict-non {{background: #FDE8EA; border-color: {ROUGE};}}
</style>
""")


def entete(droite=""):
    afficher_html(f"""
    <div class="entete">
    <div class="marque"><img src="data:image/png;base64,{charger_logo()}" alt="Bridgineers">
    <div><p class="t1">Role-play</p><p class="t2">CIM-ESUP · section anglaise</p></div></div>
    {droite}
    </div>
    """)


def carte_recruteur(texte, sous_titre="recruteur"):
    afficher_html(f"""
    <div class="carte"><div class="recruteur"><div class="avatar">MO</div>
    <div><p class="nom">Mr. Ozias <span>· {sous_titre}</span></p>
    <p class="question">{html.escape(texte)}</p></div></div></div>
    """)


def barre_progression(num_question, debut):
    segments = ""
    for i in range(1, NB_QUESTIONS + 1):
        if i < num_question:
            couleur = BLEU
        elif i == num_question:
            couleur = "#8FAEE8"
        else:
            couleur = BORDURE
        segments += f'<div style="flex:1;height:6px;border-radius:3px;background:{couleur}"></div>'
    code = f"""
    <div style="font-family:Inter,sans-serif;margin:0">
    <div style="display:flex;gap:6px;margin-bottom:8px">{segments}</div>
    <div style="display:flex;justify-content:space-between;font-size:13px;color:{GRIS}">
    <span id="chrono">⏱ 00:00</span><span>Question {num_question} / {NB_QUESTIONS}</span></div></div>
    <script>
    const debut = {int(debut * 1000)};
    function majChrono() {{
      const s = Math.max(0, Math.floor((Date.now() - debut) / 1000));
      document.getElementById("chrono").textContent =
        "⏱ " + String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
    }}
    majChrono();
    setInterval(majChrono, 1000);
    </script>
    """
    cadre(code, 46)


def cadre(code, hauteur):
    """Affiche un petit bloc HTML interactif (chrono, voix de Mr. Ozias)."""
    if hasattr(st, "iframe"):
        st.iframe(code, height=hauteur)
    else:
        import streamlit.components.v1 as components
        components.html(code, height=hauteur)


def lecteur_voix(audio, lecture_auto):
    """Le rond central : il s'anime quand Mr. Ozias parle, et on appuie dessus pour réécouter."""
    audio_b64 = base64.b64encode(audio).decode()
    barres = "".join(f'<span style="animation-delay:{i * 0.1:.1f}s"></span>' for i in range(9))
    code = f"""
    <style>
    body {{margin: 0; font-family: Inter, sans-serif; display: flex; flex-direction: column; align-items: center;}}
    .rond {{width: 92px; height: 92px; margin-top: 12px; border-radius: 50%; border: none; cursor: pointer;
      background: {BLEU}; color: #fff; display: flex; align-items: center; justify-content: center;
      box-shadow: 0 0 0 10px #E3EBFA;}}
    .ondes {{display: flex; gap: 4px; align-items: center; height: 30px; margin-top: 18px;}}
    .ondes span {{width: 4px; height: 6px; border-radius: 2px; background: #8FAEE8;}}
    body.parle .ondes span {{animation: onde 0.9s ease-in-out infinite;}}
    @keyframes onde {{0%, 100% {{height: 6px;}} 50% {{height: 28px;}}}}
    #texte {{font-size: 13px; color: {GRIS}; margin: 6px 0 0;}}
    </style>
    <button class="rond" id="bouton" aria-label="Réécouter Mr. Ozias">
    <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
    stroke-linecap="round" stroke-linejoin="round"><path d="M11 5 6 9H2v6h4l5 4V5z"/>
    <path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M19 5a10 10 0 0 1 0 14"/></svg></button>
    <div class="ondes">{barres}</div>
    <p id="texte">Appuie pour écouter Mr. Ozias</p>
    <audio id="voix" src="data:audio/mpeg;base64,{audio_b64}"></audio>
    <script>
    const voix = document.getElementById("voix");
    const texte = document.getElementById("texte");
    function etat(parle) {{
      document.body.classList.toggle("parle", parle);
      texte.textContent = parle ? "Mr. Ozias parle…" : "Appuie pour réécouter";
    }}
    voix.onplay = () => etat(true);
    voix.onended = () => etat(false);
    voix.onpause = () => etat(false);
    document.getElementById("bouton").onclick = () => {{ voix.currentTime = 0; voix.play(); }};
    if ({"true" if lecture_auto else "false"}) {{
      voix.play().catch(() => {{ texte.textContent = "Appuie pour écouter Mr. Ozias"; }});
    }}
    </script>
    """
    cadre(code, 175)


# ---------- Écran 1 : accueil ----------
def ecran_accueil():
    entete()
    afficher_html(f"""
    <div class="carte"><div class="recruteur"><div class="avatar">MO</div>
    <div><p class="nom">Mr. Ozias <span>· recruteur</span></p>
    <p class="petit">Compagnie minière internationale</p></div></div>
    <p style="margin:12px 0 0;font-size:15px">🎯 <b>Ta mission</b> : convaincre Mr. Ozias de t'embaucher,
    en anglais, en {NB_QUESTIONS} questions.</p></div>
    """)

    st.markdown("**Ton niveau d'anglais**")
    niveau = st.radio(
        "Ton niveau d'anglais", list(NIVEAUX.keys()), horizontal=True, label_visibility="collapsed",
        captions=["Mr. Ozias parle lentement et simplement", "Une conversation plus naturelle"],
    )

    st.markdown("**Avant de commencer**")
    afficher_html("""
    <div class="carte">
    <p style="margin:0 0 8px">🗣️ Fais des <b>phrases complètes</b>, pas un seul mot.</p>
    <p style="margin:0 0 8px">💡 Bloqué ? Utilise <b>l'aide</b> pour voir la traduction.</p>
    <p style="margin:0">🎧 Parle dans un <b>endroit calme</b>, près du micro.</p>
    </div>
    """)

    if st.button("Commencer l'entretien", type="primary"):
        nouvel_entretien(niveau)
        st.rerun()


# ---------- Écran 2 : entretien ----------
def envoyer(texte):
    st.session_state["historique"].append({"role": "user", "content": texte})
    st.session_state["en_attente"] = None
    with st.spinner("Mr. Ozias réfléchit..."):
        try:
            reponse = repondre(st.session_state["historique"], st.session_state["niveau"])
            st.session_state["historique"].append({"role": "assistant", "content": reponse})
            st.session_state["a_jouer"] = True
        except Exception:
            st.session_state["historique"].pop()
            st.session_state["erreur"] = "Mr. Ozias ne répond pas. Réessaie dans une minute."
    st.rerun()


def lancer_bilan():
    lignes = []
    for message in st.session_state["historique"]:
        qui = "Mr. Ozias" if message["role"] == "assistant" else "Étudiant"
        lignes.append(f"{qui} : {message['content']}")
    with st.spinner("Mr. Ozias prépare ton bilan..."):
        try:
            st.session_state["bilan"] = evaluer("\n".join(lignes), st.session_state["niveau"])
            st.session_state["etape"] = "bilan"
        except Exception:
            st.session_state["erreur"] = "Le bilan n'a pas pu être généré. Réessaie dans une minute."
    st.rerun()


def ecran_entretien():
    niveau = st.session_state["niveau"]
    historique = st.session_state["historique"]
    nb_reponses = sum(1 for message in historique if message["role"] == "user")
    termine = nb_reponses >= NB_QUESTIONS
    num_question = min(nb_reponses + 1, NB_QUESTIONS)

    entete(f'<span class="pastille">{niveau}</span>')
    barre_progression(num_question, st.session_state["debut"])

    if st.session_state.get("erreur"):
        st.error(st.session_state["erreur"])
        st.session_state["erreur"] = None

    # La dernière phrase de Mr. Ozias, avec sa voix
    derniere = historique[-1]["content"]
    try:
        lecteur_voix(parler(derniere, niveau == "Débutant"), st.session_state["a_jouer"])
    except Exception:
        pass
    st.session_state["a_jouer"] = False
    carte_recruteur(derniere)

    # Fin de l'entretien
    if termine:
        afficher_html("""
        <p class="etat">✅ L'entretien est terminé</p>
        <p class="sous-etat">Mr. Ozias a posé toutes ses questions. Découvre ton bilan !</p>
        """)
        if st.button("Voir mon bilan", key="btn_bilan"):
            lancer_bilan()
        return

    # L'aide : traduction + début de phrase
    if st.toggle("💡 Besoin d'aide ?", key=f"aide_{nb_reponses}"):
        aide = aider(derniere)
        if aide:
            afficher_html(f"""
            <div class="carte">
            <p class="petit">🇫🇷 {html.escape(aide.get("traduction", ""))}</p>
            <p style="margin:6px 0 0">Pour commencer : <b>{html.escape(aide.get("debut", ""))}</b></p>
            </div>
            """)
        else:
            st.caption("L'aide n'est pas disponible pour le moment.")

    # Vérifier ce qui a été compris
    if st.session_state["en_attente"]:
        afficher_html(f"""
        <p class="etat">👂 Vérifie ta réponse</p>
        <div class="carte"><p class="petit">J'ai entendu :</p>
        <p class="question">« {html.escape(st.session_state["en_attente"])} »</p></div>
        """)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Envoyer", type="primary"):
                envoyer(st.session_state["en_attente"])
        with col2:
            if st.button("Réenregistrer"):
                st.session_state["en_attente"] = None
                st.rerun()
    else:
        afficher_html("""
        <p class="etat">🎙️ À toi de répondre</p>
        <p class="sous-etat">Appuie sur le micro, parle en anglais, puis appuie à nouveau.</p>
        """)
        with st.container(key="zone_micro"):
            audio = st.audio_input("Ta réponse", key=f"micro_{st.session_state['num_micro']}",
                                   label_visibility="collapsed")
        if audio:
            with st.spinner("Je t'écoute..."):
                try:
                    texte = transcrire(audio)
                except Exception:
                    texte = ""
            st.session_state["num_micro"] += 1
            if texte:
                st.session_state["en_attente"] = texte
            else:
                st.session_state["erreur"] = "Je n'ai pas compris ton enregistrement. Réessaie, ou écris ta réponse."
            st.rerun()

        if st.toggle("⌨️ Écrire plutôt", key="mode_ecrit"):
            with st.form("reponse_ecrite", clear_on_submit=True):
                texte = st.text_area("Ta réponse en anglais", placeholder="I am a mining engineering student...")
                if st.form_submit_button("Envoyer"):
                    if texte.strip():
                        envoyer(texte.strip())
                    else:
                        st.warning("Écris d'abord ta réponse.")

    st.write("")
    if nb_reponses >= MIN_POUR_BILAN:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Recommencer"):
                nouvel_entretien(niveau)
                st.rerun()
        with col2:
            if st.button("Terminer · bilan", key="btn_bilan"):
                lancer_bilan()
    else:
        if st.button("Recommencer"):
            nouvel_entretien(niveau)
            st.rerun()
        restant = MIN_POUR_BILAN - nb_reponses
        st.caption(f"Encore {restant} réponse{'s' if restant > 1 else ''} avant de pouvoir demander ton bilan.")


# ---------- Écran 3 : bilan ----------
def liste_html(elements):
    return "<ul>" + "".join(f"<li>{html.escape(str(e))}</li>" for e in elements) + "</ul>"


def corrections_html(elements):
    code = ""
    for e in elements:
        code += (f'<div class="correction">Tu as dit : <span class="dit">{html.escape(str(e.get("dit", "")))}</span><br>'
                 f'Dis plutôt : <span class="mieux">{html.escape(str(e.get("mieux", "")))}</span><br>'
                 f'<span class="petit">{html.escape(str(e.get("pourquoi", "")))}</span></div>')
    return code


def ecran_bilan():
    entete(f'<span class="pastille">{st.session_state["niveau"]}</span>')
    bilan = st.session_state["bilan"]

    if "brut" in bilan:
        st.subheader("Ton bilan d'entretien")
        st.markdown(bilan["brut"])
    else:
        accepte = str(bilan.get("decision", "")).lower().startswith("accept")
        if accepte:
            titre, classe = "🎉 Félicitations, tu es embauché !", "verdict-ok"
            if not st.session_state["ballons"]:
                st.balloons()
                st.session_state["ballons"] = True
        else:
            titre, classe = "Mr. Ozias ne t'a pas retenu cette fois", "verdict-non"
        afficher_html(f"""
        <div class="carte {classe}"><h4>{titre}</h4>
        <p style="margin:0">{html.escape(str(bilan.get("explication", "")))}</p></div>
        """)

        # Notes par critère
        barres = ""
        for cle, nom in [("anglais", "Anglais"), ("contenu", "Contenu des réponses"), ("assurance", "Assurance")]:
            try:
                note = max(0, min(10, int(bilan.get("notes", {}).get(cle, 0))))
            except Exception:
                note = 0
            barres += (f'<div class="note"><span>{nom}</span><b>{note} / 10</b></div>'
                       f'<div class="barre"><div style="width:{note * 10}%"></div></div>')
        afficher_html(f'<div class="carte"><h4>📊 Tes notes</h4>{barres}</div>')

        if bilan.get("points_forts"):
            afficher_html(f'<div class="carte"><h4>✅ Points forts</h4>{liste_html(bilan["points_forts"])}</div>')

        corrections = bilan.get("corrections") or []
        if corrections:
            afficher_html(f'<div class="carte"><h4>✏️ À corriger</h4>{corrections_html(corrections)}</div>')
        else:
            afficher_html('<div class="carte"><h4>✏️ À corriger</h4><p class="petit">Aucune faute relevée, bravo !</p></div>')

        with st.expander("Voir plus de conseils"):
            if bilan.get("style"):
                afficher_html(f'<div class="carte"><h4>✨ Pour sonner plus naturel</h4>{corrections_html(bilan["style"])}</div>')
            if bilan.get("conseils"):
                afficher_html(f'<div class="carte"><h4>🎯 Conseils pour un vrai entretien</h4>{liste_html(bilan["conseils"])}</div>')
            if bilan.get("phrases_utiles"):
                afficher_html(f'<div class="carte"><h4>💬 Phrases utiles</h4>{liste_html(bilan["phrases_utiles"])}</div>')

    with st.expander("Revoir l'entretien"):
        for message in st.session_state["historique"]:
            qui = "👔 Mr. Ozias" if message["role"] == "assistant" else "⛏️ Toi"
            st.markdown(f"**{qui}** : {message['content']}")

    if st.button("🔁 Retenter l'entretien", type="primary"):
        nouvel_entretien(st.session_state["niveau"])
        st.rerun()
    if st.button("Changer de niveau"):
        retour_accueil()
        st.rerun()
    st.caption("Bridgineers · CIM-ESUP")


# ---------- Lancement ----------
if st.session_state["etape"] == "accueil":
    ecran_accueil()
elif st.session_state["etape"] == "entretien":
    ecran_entretien()
else:
    ecran_bilan()
