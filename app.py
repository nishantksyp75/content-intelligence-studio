from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION

import tempfile
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

import faiss
import gradio as gr
import whisper
from huggingface_hub import hf_hub_download
from keybert import KeyBERT
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading Whisper...")
whisper_model = whisper.load_model("base")

print("Loading TextRank...")
text_rank = TextRankSummarizer()

print("Loading KeyBERT...")
kw_model = KeyBERT()

print("Loading sentence embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_PATH = None
llm = None

def groq_chat(messages, max_completion_tokens=500, temperature=0.2):
    """Call Groq's OpenAI-compatible chat completions API."""
    import json
    import urllib.request
    import urllib.error

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
    }

    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"Groq API returned HTTP {error.code}: {details}"
        ) from error

    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError("Groq returned no completion choices.")

    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Groq returned an empty response.")

    return content.strip()


def get_llm():
    global MODEL_PATH, llm
    if llm is None:
        print("Locating Qwen model...")
        MODEL_PATH = hf_hub_download(repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF", filename="qwen2.5-1.5b-instruct-q4_k_m.gguf")
        print("Loading local Qwen model...")
        llm = Llama(model_path=MODEL_PATH, n_ctx=1024, n_threads=2, verbose=False)
    return llm

print("Other models loaded successfully!")


# ============================================================
# TRANSCRIPT CORRECTION
# ============================================================


def split_into_groups(text, max_chars=1000):
    """Split text into manageable sections while preserving content."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    groups, current_group, current_length = [], [], 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current_group:
                groups.append(" ".join(current_group))
                current_group, current_length = [], 0

            groups.extend(
                sentence[start : start + max_chars]
                for start in range(0, len(sentence), max_chars)
            )
            continue

        additional_length = len(sentence) + (1 if current_group else 0)

        if current_group and current_length + additional_length > max_chars:
            groups.append(" ".join(current_group))
            current_group, current_length = [], 0

        current_group.append(sentence)
        current_length += len(sentence) + (1 if len(current_group) > 1 else 0)

    if current_group:
        groups.append(" ".join(current_group))

    return groups


def correct_transcription(text):
    """Use Qwen to correct likely speech-recognition errors."""
    if not text or not text.strip():
        return text

    groups = split_into_groups(text)
    corrected_groups = []

    system_prompt = (
        "You are an expert speech-transcription correction editor "
        "specializing in technical content. Correct likely speech "
        "recognition errors using context. Pay special attention to "
        "programming languages, Python libraries, frameworks, software "
        "tools, and technical terms that may be transcribed phonetically. "
        "Use standard spellings when the surrounding sentence strongly "
        "supports the intended term. For example, 'Numpi' may mean "
        "'NumPy', 'tensor flow' may mean 'TensorFlow', 'psychic learn' "
        "may mean 'scikit-learn', and 'Keev' may mean 'Kivy' in the "
        "appropriate context. Do not replace words blindly. Preserve "
        "the original meaning, wording, order, names, dates, times, "
        "numbers, and facts. Do not summarize, paraphrase, or add "
        "information. If uncertain, leave the wording unchanged. "
        "Return only the corrected transcript without explanations."
    )

    for number, group in enumerate(groups, start=1):
        print(f"Correcting transcript section {number}/{len(groups)}...")

        try:
            response = get_llm().create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": group},
                ],
                max_tokens=900,
                temperature=0.0,
            )

            choices = response.get("choices", [])
            corrected = (
                choices[0].get("message", {}).get("content", "").strip()
                if choices
                else ""
            )

            if not corrected:
                print("Empty correction; keeping original section.")
                corrected_groups.append(group)

            elif len(corrected) < len(group) * 0.55:
                print("Correction too short; keeping original section.")
                corrected_groups.append(group)

            elif len(corrected) > len(group) * 1.6:
                print("Correction too long; keeping original section.")
                corrected_groups.append(group)

            else:
                corrected_groups.append(corrected)

        except Exception as error:
            print("Transcript correction error:", repr(error))
            corrected_groups.append(group)

    return " ".join(corrected_groups)


# ============================================================
# FAISS KNOWLEDGE BASE
# ============================================================

chunks = []
faiss_index = None


def create_rag_index(transcript):
    """Build a semantic-search index from transcript chunks."""
    global chunks, faiss_index

    if not transcript or not transcript.strip():
        chunks = []
        faiss_index = None
        raise gr.Error("The transcript is empty.")

    sentences = re.split(
        r"(?<=[.!?])\s+",
        transcript.replace("\n", " ").strip(),
    )
    sentences = [s.strip() for s in sentences if s.strip()]

    chunk_size, overlap = 5, 2
    step = chunk_size - overlap
    new_chunks = []

    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start : start + chunk_size]).strip()
        if chunk:
            new_chunks.append(chunk)

    if not new_chunks:
        faiss_index = None
        chunks = []
        raise gr.Error("Could not create searchable chunks from the transcript.")

    print("Number of RAG chunks:", len(new_chunks))

    embeddings = embedding_model.encode(
        new_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Replace the active index only after successful construction.
    chunks = new_chunks
    faiss_index = index

    print("FAISS index created. Vectors:", faiss_index.ntotal)


def retrieve_context(question, k=5):
    """Retrieve the most relevant transcript chunks."""
    if faiss_index is None or not chunks:
        return ""

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    count = min(k, faiss_index.ntotal)
    if count <= 0:
        return ""

    _, indices = faiss_index.search(question_embedding, count)

    retrieved = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]

    return "\n\n".join(dict.fromkeys(retrieved))


# ============================================================
# AUDIO AND VIDEO ANALYSIS
# ============================================================


def analyze_media(media_file):
    if media_file is None:
        raise gr.Error("Please upload an audio or video file.")

    extension = os.path.splitext(media_file)[1].lower()

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}

    if extension not in video_extensions | audio_extensions:
        raise gr.Error(f"Unsupported file type: {extension or 'unknown'}")

    temporary_audio_path = None

    try:
        # Extract audio from videos using FFmpeg.
        if extension in video_extensions:
            print("Video detected. Extracting audio...")

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temporary_audio:
                temporary_audio_path = temporary_audio.name

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    media_file,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    temporary_audio_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            audio_file = temporary_audio_path

        else:
            audio_file = media_file

        # Transcribe with Whisper.
        print("Transcribing with Whisper...")
        result = whisper_model.transcribe(audio_file, fp16=False)
        original_text = result.get("text", "").strip()

        if not original_text:
            raise gr.Error("No speech could be detected in the uploaded file.")

        # Correct likely transcription errors with Qwen.
        print("Correcting transcript...")
        print("Skipping slow Qwen transcript correction; using Whisper transcript.")
        corrected_text = original_text.strip()

        if not corrected_text:
            corrected_text = original_text

        # Generate an abstractive summary with Groq.
        print("Generating human-readable summary with Groq...")

        try:
            # Bound the input size to keep requests manageable.
            transcript_for_summary = corrected_text[:40000]

            summary = groq_chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert content analyst. Create an "
                            "accurate, useful abstractive summary of the "
                            "provided transcript. Start with the main message, "
                            "combine related ideas into coherent paragraphs, "
                            "remove repetition, and preserve important facts, "
                            "names, numbers, and recommendations. Do not invent "
                            "information. Return only the summary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Summarize this transcript:\n\n"
                            + transcript_for_summary
                        ),
                    },
                ],
                max_completion_tokens=700,
                temperature=0.2,
            )

        except Exception as e:
            print(f"Groq summarization failed: {e}")
            print("Falling back to TextRank...")

            try:
                parser = PlaintextParser.from_string(
                    corrected_text,
                    Tokenizer("english"),
                )
                summary_sentences = text_rank(parser.document, 5)
                summary = "\n\n".join(
                    str(sentence) for sentence in summary_sentences
                )
            except Exception as fallback_error:
                print(f"TextRank fallback failed: {fallback_error}")
                summary = ""

            if not summary:
                summary = "The transcript was too short to generate a summary."

        # Extract meaningful keywords.
        print("Extracting meaningful keywords...")

        raw_keywords = kw_model.extract_keywords(
            corrected_text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=30,
            use_mmr=True,
            diversity=0.6,
        )

        generic_terms = {
            "learn",
            "learning",
            "basics",
            "things",
            "work",
            "actually",
            "everything",
            "understand",
            "going",
            "start",
            "use",
            "need",
            "know",
            "made",
            "teach",
            "course",
            "youtube",
            "really",
            "first",
            "second",
            "finally",
        }

        keywords = []
        seen = set()

        for phrase, score in raw_keywords:
            normalized = " ".join(phrase.lower().split())

            if not normalized or normalized in seen:
                continue

            words = normalized.split()

            if all(word in generic_terms for word in words):
                continue

            seen.add(normalized)
            keywords.append((phrase, float(score)))

            if len(keywords) == 10:
                break

        keyword_text = (
            "\n".join(
                f"{phrase} (relevance: {score:.3f})" for phrase, score in keywords
            )
            if keywords
            else "No meaningful keywords could be extracted."
        )

        # Build the semantic-search knowledge base.
        print("Creating FAISS search index...")
        create_rag_index(corrected_text)

        print("Analysis complete.")
        return corrected_text, summary, keyword_text

    except subprocess.CalledProcessError as error:
        print("FFmpeg error:", error.stderr or str(error))

        raise gr.Error(
            "Could not extract audio from the video. "
            "Check that FFmpeg is installed and that the file is valid."
        )

    except gr.Error:
        raise

    except Exception as error:
        print("Analysis error:", repr(error))

        raise gr.Error("Analysis failed. Check the terminal for the detailed error.")

    finally:
        if temporary_audio_path and os.path.exists(temporary_audio_path):
            try:
                os.remove(temporary_audio_path)
                print("Temporary audio file deleted.")
            except OSError as error:
                print("Could not delete temporary audio:", repr(error))


# ============================================================
# QUESTION ANSWERING
# ============================================================


def ask_question(question):
    if not question or not question.strip():
        return "Please enter a question."

    if faiss_index is None or not chunks:
        return "Please analyze an audio or video file first."

    question = question.strip()
    print("Question:", question)

    # Support numbered lists of questions as well as a single question.
    questions = re.split(r"(?m)^\s*\d+[.)]\s*", question)
    questions = [q.strip() for q in questions if q.strip()]
    if len(questions) <= 1:
        questions = [question]

    answers = []

    system_prompt = (
        "You answer questions about an audio or video transcript. "
        "Use only the supplied transcript context as factual evidence. "
        "Answer the user's question directly and concisely. If asked to "
        "generate questions, provide actual questions rather than answers. "
        "Do not invent facts or introduce unsupported topics. If the "
        "context is insufficient, say so."
    )

    for number, current_question in enumerate(questions, start=1):
        print(f"Processing question {number}: {current_question}")

        try:
            context = retrieve_context(current_question, k=5)

            if not context.strip():
                answer = (
                    "I could not find relevant information in the transcript."
                )
            else:
                answer = groq_chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Transcript context:\n{context}\n\n"
                                f"Question: {current_question}\n\n"
                                "Answer using only the supplied context."
                            ),
                        },
                    ],
                    max_completion_tokens=400,
                    temperature=0.1,
                )

            answers.append(f"{number}. {current_question}\n{answer}")
            print(f"Answer {number} generated successfully.")

        except Exception as error:
            print(
                f"Question-answering error for question {number}:",
                repr(error),
            )
            answers.append(
                f"{number}. {current_question}\n"
                "Answer generation failed. Please try again."
            )

    return "\n\n".join(answers)


# ============================================================
# FORMATTED WORD REPORT EXPORT
# ============================================================


def export_report_docx(transcript, summary, keywords, qa_history=""):
    doc = Document()

    # Page layout.
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    # Default body font.
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(11)

    # Helper for section headings.
    def add_heading(text):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_before = Pt(12)
        paragraph.paragraph_format.space_after = Pt(6)

        run = paragraph.add_run(text)
        run.font.name = "Helvetica"
        run.font.size = Pt(14)
        run.bold = True

    # Main title.
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(4)

    run = title.add_run("CONTENT INTELLIGENCE STUDIO")
    run.font.name = "Helvetica"
    run.font.size = Pt(18)
    run.bold = True

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    run = subtitle.add_run("AI Audio & Video Analysis Report")
    run.font.name = "Arial"
    run.font.size = Pt(11)

    generated = doc.add_paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    generated.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Summary.
    add_heading("1. SUMMARY")
    doc.add_paragraph(str(summary or "No summary available."))

    # Keywords.
    add_heading("2. KEYWORDS")

    if isinstance(keywords, (list, tuple)):
        for keyword in keywords:
            doc.add_paragraph(str(keyword), style="List Bullet")
    else:
        doc.add_paragraph(str(keywords or "No keywords available."))

    # Transcript.
    add_heading("3. TRANSCRIPT")
    doc.add_paragraph(str(transcript or "No transcript available."))

    # Questions and answers.
    add_heading("4. QUESTIONS & ANSWERS")

    if qa_history and qa_history.strip():
        for line in qa_history.splitlines():
            paragraph = doc.add_paragraph()

            if line.strip().lower().startswith("question:"):
                run = paragraph.add_run(line)
                run.bold = True
            else:
                paragraph.add_run(line)
    else:
        doc.add_paragraph("No questions have been answered yet.")

    # Save to a temporary DOCX file.
    output_path = os.path.join(
        tempfile.gettempdir(),
        "content_intelligence_report.docx",
    )

    doc.save(output_path)

    return output_path


# ============================================================
# GRADIO INTERFACE
# ============================================================

custom_css = """
.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}

#app-header {
    padding: 28px 30px;
    border-radius: 18px;
    background: linear-gradient(120deg, #101a35, #243f73);
    color: white;
    margin-bottom: 20px;
}

#app-header h1 {
    color: white;
    font-size: 30px;
    margin-bottom: 8px;
}

#app-header p {
    color: #dbe6ff;
    font-size: 14px;
}

.primary-btn {
    background: #356df3 !important;
    color: white !important;
    border: none !important;
}

.primary-btn:hover {
    background: #2457d5 !important;
}

#transcript-box textarea,
#summary-box textarea,
#keywords-box textarea {
    height: 260px !important;
    max-height: 260px !important;
    overflow-y: scroll !important;
    resize: vertical !important;
}

#transcript-box,
#summary-box,
#keywords-box {
    min-height: 0 !important;
}

footer {
    display: none !important;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    css=custom_css,
    title="Content Intelligence Studio",
) as app:

    qa_history = gr.State("")

    gr.HTML("""
    <div id="app-header">
        <h1>Content Intelligence Studio</h1>
        <p>Transform audio and video into searchable knowledge using AI models.</p>
    </div>
    """)

    # Upload and analyze.
    with gr.Group():
        gr.Markdown("## 01 · Upload and analyze")

        media_file = gr.File(
            label="Choose an audio or video file",
            file_types=[
                ".mp3",
                ".wav",
                ".m4a",
                ".flac",
                ".aac",
                ".ogg",
                ".mp4",
                ".mov",
                ".avi",
                ".mkv",
                ".webm",
            ],
            type="filepath",
        )

        analyze_button = gr.Button(
            "Analyze media",
            variant="primary",
            elem_classes=["primary-btn"],
        )

        gr.Markdown(
            "Supports transcription, video audio extraction, summarization, "
            "keyword extraction, and semantic retrieval."
        )

    # Analysis results.
    gr.Markdown("## 02 · Analysis results")

    with gr.Tabs():
        with gr.Tab("Transcript"):
            transcript_output = gr.Textbox(
                label="Transcript",
                placeholder="Your transcript will appear here after analysis.",
                lines=10,
                interactive=False,
                elem_id="transcript-box",
            )

        with gr.Tab("Summary"):
            summary_output = gr.Textbox(
                label="Abstractive summary",
                placeholder="The key points will appear here.",
                lines=8,
                interactive=False,
                elem_id="summary-box",
            )

        with gr.Tab("Keywords"):
            keywords_output = gr.Textbox(
                label="Extracted keywords",
                placeholder="Important terms will appear here.",
                lines=6,
                interactive=False,
                elem_id="keywords-box",
            )

    # Question and answer.
    gr.Markdown("## 03 · Ask your content")

    gr.Markdown(
        "Ask a question about the uploaded media. Answers use relevant "
        "transcript sections retrieved from the knowledge base."
    )

    question_input = gr.Textbox(
        label="Your question",
        placeholder="Example: What are the main ideas discussed in this video?",
        lines=3,
    )

    ask_button = gr.Button(
        "Generate answer",
        variant="primary",
        elem_classes=["primary-btn"],
    )

    answer_output = gr.Textbox(
        label="AI-generated answer",
        placeholder="Your answer will appear here.",
        lines=8,
        interactive=False,
    )

    # Preserve Q&A history for report export.
    def answer_and_record(question, history):
        answer = ask_question(question)
        history = history or ""

        if question and question.strip():
            history += f"Question: {question.strip()}\n" f"Answer: {answer}\n\n"

        return answer, history

    analysis_status = gr.Markdown("")
    answer_status = gr.Markdown("")
    docx_status = gr.Markdown("")

    # Export report components
    gr.Markdown("## 04 · Export report")
    gr.Markdown(
        "Download the transcript, summary, keywords, and recorded Q&A "
        "as a formatted Word document."
    )
    docx_button = gr.Button(
        "Export Word Report",
        variant="primary",
        elem_classes=["primary-btn"],
    )
    docx_file = gr.File(
        label="Download Word report",
        interactive=False,
    )

    # Each action locks only its own button until another file is uploaded.
    def upload_unlock(media_path):
        if media_path is None:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
        return (
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(interactive=True, value="Analyze media"),
            gr.update(interactive=True, value="Generate answer"),
            gr.update(interactive=True, value="Export Word Report"),
        )

    def analysis_started():
        return (
            gr.update(value="⏳ **Analyzing media... Please wait.**"),
            gr.update(interactive=False, value="Analyze media"),
            gr.update(),
            gr.update(),
        )

    def analysis_done(transcript, summary, keywords):
        return (
            transcript,
            summary,
            keywords,
            gr.update(value="✅ **Analysis complete. Upload a new file to enable Analyze media again.**"),
            gr.update(interactive=False, value="Analyze media"),
            gr.update(),
            gr.update(),
        )

    def analysis_failed():
        return (
            gr.update(value="❌ **Analysis failed. Check server logs. Upload a new file to enable Analyze media again.**"),
            gr.update(interactive=False, value="Analyze media"),
            gr.update(),
            gr.update(),
        )

    def answer_started():
        return (
            gr.update(value="⏳ **Generating answer... Please wait.**"),
            gr.update(),
            gr.update(interactive=False, value="Generate answer"),
            gr.update(),
        )

    def answer_done(answer, history):
        return (
            answer,
            history,
            gr.update(value="✅ **Answer generated successfully. You can ask another question.**"),
            gr.update(),
            gr.update(interactive=True, value="Generate answer"),
            gr.update(),
        )

    def answer_failed():
        return (
            "Answer generation failed. Please try again after uploading a new file.",
            gr.update(value="❌ **Answer generation failed. Check server logs.**"),
            gr.update(),
            gr.update(interactive=True, value="Generate answer"),
            gr.update(),
        )

    def export_started():
        return (
            gr.update(value="⏳ **Generating Word report... Please wait.**"),
            gr.update(),
            gr.update(),
            gr.update(interactive=False, value="Export Word Report"),
        )

    def export_done():
        return (
            gr.update(value="✅ **Report exported successfully. You can export again anytime.**"),
            gr.update(),
            gr.update(),
            gr.update(interactive=True, value="Export Word Report"),
        )

    def export_failed():
        return (
            gr.update(value="❌ **Report export failed. Check server logs. You can retry.**"),
            gr.update(),
            gr.update(),
            gr.update(interactive=True, value="Export Word Report"),
        )

    # Uploading/re-uploading a file unlocks all three buttons.
    media_file.change(
        fn=upload_unlock,
        inputs=[media_file],
        outputs=[
            analysis_status,
            answer_status,
            docx_status,
            analyze_button,
            ask_button,
            docx_button,
        ],
        queue=False,
    )

    analysis_event = analyze_button.click(
        fn=analysis_started,
        inputs=[],
        outputs=[analysis_status, analyze_button, ask_button, docx_button],
        queue=False,
        trigger_mode="once",
    ).then(
        fn=analyze_media,
        inputs=[media_file],
        outputs=[transcript_output, summary_output, keywords_output],
    ).then(
        fn=analysis_done,
        inputs=[transcript_output, summary_output, keywords_output],
        outputs=[
            transcript_output,
            summary_output,
            keywords_output,
            analysis_status,
            analyze_button,
            ask_button,
            docx_button,
        ],
    ).failure(
        fn=analysis_failed,
        inputs=[],
        outputs=[analysis_status, analyze_button, ask_button, docx_button],
    )

    ask_event = ask_button.click(
        fn=answer_started,
        inputs=[],
        outputs=[answer_status, analyze_button, ask_button, docx_button],
        queue=False,
        trigger_mode="once",
    ).then(
        fn=answer_and_record,
        inputs=[question_input, qa_history],
        outputs=[answer_output, qa_history],
    ).then(
        fn=answer_done,
        inputs=[answer_output, qa_history],
        outputs=[
            answer_output,
            qa_history,
            answer_status,
            analyze_button,
            ask_button,
            docx_button,
        ],
    ).failure(
        fn=answer_failed,
        inputs=[],
        outputs=[
            answer_output,
            answer_status,
            analyze_button,
            ask_button,
            docx_button,
        ],
    )

    # Enter key follows the same independent Generate Answer lock.
    question_input.submit(
        fn=answer_started,
        inputs=[],
        outputs=[answer_status, analyze_button, ask_button, docx_button],
        queue=False,
        trigger_mode="once",
    ).then(
        fn=answer_and_record,
        inputs=[question_input, qa_history],
        outputs=[answer_output, qa_history],
    ).then(
        fn=answer_done,
        inputs=[answer_output, qa_history],
        outputs=[
            answer_output,
            qa_history,
            answer_status,
            analyze_button,
            ask_button,
            docx_button,
        ],
    ).failure(
        fn=answer_failed,
        inputs=[],
        outputs=[
            answer_output,
            answer_status,
            analyze_button,
            ask_button,
            docx_button,
        ],
    )

    docx_button.click(
        fn=export_started,
        inputs=[],
        outputs=[docx_status, analyze_button, ask_button, docx_button],
        queue=False,
        trigger_mode="once",
    ).then(
        fn=export_report_docx,
        inputs=[
            transcript_output,
            summary_output,
            keywords_output,
            qa_history,
        ],
        outputs=[docx_file],
    ).then(
        fn=export_done,
        inputs=[],
        outputs=[docx_status, analyze_button, ask_button, docx_button],
    ).failure(
        fn=export_failed,
        inputs=[],
        outputs=[docx_status, analyze_button, ask_button, docx_button],
    )


if __name__ == "__main__":
    app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    share=False,
    )
