import gradio as gr
import requests
import json

API_URL = "http://localhost:8000/api"

# Helper functions for project and profile management

def list_profiles():
    try:
        resp = requests.get(f"{API_URL}/profile/list")
        return resp.json().get("profiles", [])
    except Exception:
        return []

def list_projects():
    try:
        resp = requests.get(f"{API_URL}/project/list")
        return resp.json().get("projects", [])
    except Exception:
        return []

def create_project(name, script, speakers):
    try:
        resp = requests.post(f"{API_URL}/project/create", data={"name": name, "script": script, "speakers": json.dumps(speakers)})
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def generate_project_audio(name):
    try:
        resp = requests.post(f"{API_URL}/project/generate_audio", data={"name": name})
        return resp.json().get("audio_path", None)
    except Exception:
        return None

def build_multivoice_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Multi-Voice Project Studio")
        with gr.Tab("Create Project"):
            name = gr.Textbox(label="Project Name")
            script = gr.Textbox(label="Script (Speaker: line)", lines=8)
            profile_list = gr.Dropdown(choices=list_profiles(), label="Available Profiles", multiselect=True)
            speakers = gr.Textbox(label="Speaker Mapping (JSON, e.g. {'A':'profile1','B':'profile2'})")
            create_btn = gr.Button("Create Project")
            create_out = gr.Textbox(label="Status")
            def create_proj(name, script, mapping):
                try:
                    speakers = json.loads(mapping)
                except Exception:
                    return "Invalid speaker mapping JSON"
                result = create_project(name, script, speakers)
                return str(result)
            create_btn.click(fn=create_proj, inputs=[name, script, speakers], outputs=create_out)
        with gr.Tab("Generate Audio"):
            proj_list = gr.Dropdown(choices=list_projects(), label="Select Project")
            gen_btn = gr.Button("Generate Audio")
            audio_out = gr.Audio(label="Project Audio")
            def gen_audio(name):
                path = generate_project_audio(name)
                return path
            gen_btn.click(fn=gen_audio, inputs=proj_list, outputs=audio_out)
    return demo

if __name__ == "__main__":
    ui = build_multivoice_ui()
    ui.launch(server_port=7861)
