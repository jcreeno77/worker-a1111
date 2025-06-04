import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry

BASE_URL = "http://127.0.0.1:3000"
SD_API_URL = f"{BASE_URL}/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    """
    Check if the service is ready to receive requests.
    """
    retries = 0

    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            retries += 1

            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)


def run_inference(inference_request):
    """Run txt2img inference."""
    response = automatic_session.post(
        url=f"{SD_API_URL}/txt2img",
        json=inference_request,
        timeout=600,
    )
    if not response.ok:
        raise RuntimeError(response.text)
    return response.json()


def run_birefnet_single(biref_request):
    """Forward a request to the BiRefNet single endpoint."""
    response = automatic_session.post(
        url=f"{BASE_URL}/birefnet/single",
        json=biref_request,
        timeout=600,
    )
    if not response.ok:
        raise RuntimeError(response.text)
    return response.json()


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    """Dispatch requests to Automatic1111 or BiRefNet."""
    action = event.get("action", "txt2img")
    payload = event.get("input", {})

    if action == "birefnet_single":
        return run_birefnet_single(payload)

    return run_inference(payload)


if __name__ == "__main__":
    wait_for_service(url=f"{SD_API_URL}/sd-models")
    print("WebUI API Service is ready. Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})
