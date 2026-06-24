"""Captura 5 pantallas de la app desplegada con Playwright."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://JoeUgalde40-ecg-cnn1d-api.hf.space"
OUT = Path("doc_assets"); OUT.mkdir(exist_ok=True)
MULTI_ID = "146"
NORM_ID = "13"


def classify_and_shot(page, ecg_id, outfile, gradcam=False):
    page.goto(BASE + "/", wait_until="networkidle")
    page.wait_for_function("document.querySelectorAll('#sel option').length > 0", timeout=30000)
    page.select_option("#sel", ecg_id)
    page.click("text=Clasificar ECG")
    # esperar a que aparezca el diagnóstico y la gráfica
    page.wait_for_function("document.getElementById('out').innerHTML.includes('Diagn')", timeout=40000)
    page.wait_for_function(
        "(() => {const i=document.getElementById('plot'); return i && i.complete && i.naturalWidth>0;})()",
        timeout=40000)
    page.wait_for_timeout(1200)
    if gradcam:
        page.click("#gbtn")
        page.wait_for_function(
            "(() => {const i=document.getElementById('gcam'); return i && i.complete && i.naturalWidth>0;})()",
            timeout=60000)
        page.wait_for_timeout(1500)
    page.screenshot(path=str(outfile), full_page=True)
    print("OK", outfile)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)

    # --- Captura 1: home ---
    page.goto(BASE + "/", wait_until="networkidle")
    page.wait_for_function("document.querySelectorAll('#sel option').length > 0", timeout=30000)
    page.wait_for_timeout(800)
    page.screenshot(path=str(OUT / "cap1.png"), full_page=True)
    print("OK cap1 (home)")

    # --- Captura 2: multi-etiqueta ---
    classify_and_shot(page, MULTI_ID, OUT / "cap2.png")

    # --- Captura 3: NORM ---
    classify_and_shot(page, NORM_ID, OUT / "cap3.png")

    # --- Captura 6: Grad-CAM en la web (clasifica #274 y muestra el mapa de calor) ---
    classify_and_shot(page, "274", OUT / "cap6.png", gradcam=True)

    # --- Captura 4: Swagger /docs ---
    page.goto(BASE + "/docs", wait_until="networkidle")
    page.wait_for_selector(".opblock", timeout=30000)
    page.wait_for_timeout(1000)
    page.screenshot(path=str(OUT / "cap4.png"), full_page=True)
    print("OK cap4 (swagger)")

    # --- Captura 5: ejecutar POST /predict en Swagger ---
    try:
        # expandir el bloque POST /predict
        post = page.locator(".opblock-post", has=page.locator("text=/predict")).first
        post.locator(".opblock-summary").click()
        page.wait_for_timeout(600)
        page.locator(".opblock-post .try-out__btn").first.click()
        page.wait_for_timeout(400)
        ta = page.locator(".opblock-post textarea").first
        ta.fill('{\n  "ecg_id": 146\n}')
        page.locator(".opblock-post .execute").first.click()
        page.wait_for_selector(".opblock-post .responses-table .response .highlight-code", timeout=40000)
        page.wait_for_timeout(1200)
        post.scroll_into_view_if_needed()
        post.screenshot(path=str(OUT / "cap5.png"))
        print("OK cap5 (predict ejecutado)")
    except Exception as e:
        print("WARN cap5 falló:", e)
        page.screenshot(path=str(OUT / "cap5.png"), full_page=True)

    browser.close()
print("Listo.")
