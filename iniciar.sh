#!/usr/bin/env bash
# Enciende el servidor de la página + ngrok y muestra el link público (y un QR) para abrirla en cualquier celular.
# Para apagar todo: Ctrl + C
cd "$(dirname "$0")"

PUERTO=8080

if ! command -v ngrok >/dev/null; then
  echo "No está instalado ngrok."
  exit 1
fi

#Si quedó algo encendido de una vez anterior (por ejemplo con Ctrl+Z), apagarlo
for viejo in $(pgrep -f "http.server $PUERTO") $(pgrep -f "ngrok http $PUERTO"); do
  kill -CONT "$viejo" 2>/dev/null; kill "$viejo" 2>/dev/null
done
sleep 1

#Al presionar Ctrl+C se apagan el servidor y ngrok
trap 'echo; echo "Apagando..."; kill $SERVIDOR $TUNEL 2>/dev/null; exit 0' INT TERM

#1) Servidor de la página (solo accesible desde esta laptop; ngrok es quien lo publica)
python3 -m http.server "$PUERTO" --bind 127.0.0.1 > /dev/null 2>&1 &
SERVIDOR=$!

#2) Túnel de ngrok
ngrok http "$PUERTO" --log stdout > /tmp/ngrok_perros_gatos.log 2>&1 &
TUNEL=$!

#3) Esperar a que ngrok dé el link público
URL=""
for _ in $(seq 1 30); do
  URL=$(curl -s http://127.0.0.1:4040/api/tunnels | python3 -c "import json,sys; t=json.load(sys.stdin)['tunnels']; print(t[0]['public_url'] if t else '')" 2>/dev/null)
  [ -n "$URL" ] && break
  sleep 0.5
done

if [ -z "$URL" ]; then
  echo "ngrok no pudo iniciar. Esto dice su registro:"
  tail -5 /tmp/ngrok_perros_gatos.log
  kill $SERVIDOR $TUNEL 2>/dev/null
  exit 1
fi

echo
echo "  Abre en cualquier celular (con Wi-Fi o datos):  $URL"
echo
command -v qrencode >/dev/null && qrencode -t ansiutf8 "$URL"
echo "  Para apagar todo: Ctrl + C"
echo

wait
