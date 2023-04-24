# Start nginx
nginx
# start sshd
/usr/sbin/sshd
# Start uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 10
