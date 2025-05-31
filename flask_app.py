from flask import Flask, render_template, request, redirect, url_for, flash
import subprocess
import threading
import secrets
import signal

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a real secret

visionai_process = {"proc": None}

def run_visionai(mode):
    proc = subprocess.Popen(
        ["python", "gemini_vision.py", "--mode", mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr with stdout
        text=True
    )
    visionai_process["proc"] = proc
    # Print output for debugging
    for line in proc.stdout:
        print("[VisionAI]", line, end="")
    proc.wait()

def stop_visionai():
    proc = visionai_process.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        visionai_process["proc"] = None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "start":
            mode = request.form.get("mode", "camera")
            # Start the script in a background thread so Flask stays responsive
            threading.Thread(target=run_visionai, args=(mode,), daemon=True).start()
            flash(f"VisionAI started in {mode} mode.")
        elif action == "stop":
            stop_visionai()
            flash("VisionAI stopped.")
        return redirect(url_for("index"))
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, threaded=True)