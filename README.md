# DeepSwing
Hello, welcome to my repository where I am developing a LLM powered golf coach, called DeepSwing. DeepSwing is being designed to allow for on demand, personalized swing feedback for the average golfer. With the rapid growth of the golf community in recent years, not everyone has the money for expensive golf coaches, or the time to self-diagnose and scour the internet for advice. DeepSwing aims to solve these problems.

My YouTube channel, where DeepSwing demos will be posted until I manage to get everything up and running on a public website.

## Tech Stack
### Back-End
1. [SwingNet](https://github.com/wmcnally/golfdb?tab=readme-ov-file) - model presented in [McNally et al.](https://arxiv.org/pdf/1903.06528) with the goal of identifying time points where golf swing events occur, such as address, mid-backswing, impact, etc.

2. [Ultralytics YOLOv11](https://docs.ultralytics.com/models/yolo11/#models) - SOTA vision model used for extracting pose estimation keypoints at each of the golf swing event time points identified by SwingNet.

3. [Gemma3-4b](https://www.ollama.com/library/gemma3) - Google's current "most capable model (LLM) that runs on a single GPU", in its 4 billion parameter variant for balancing performance and speed, paired with the Ollama API.

4. [FastAPI](https://fastapi.tiangolo.com/) - Used for web app API functionality to connect python scripts for model inference to front-end web page.

5. [Docker](https://www.docker.com/) - Docker is being used to remove system requirement and depenency-based accessibility issues by containerizing all of the dependencies required for the use of each model.

### Front-End
The front end for this application is being developed with standard HTML, CSS, and JavaScript. Front-end is a new skill for me, so bear with me as I continue to develop the UI for DeepSwing :). I am hoping to host DeepSwing for public use in the near future.