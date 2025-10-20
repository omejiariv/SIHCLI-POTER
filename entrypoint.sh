#!/bin/bash
echo fs.inotify.max_user_watches=524288 | tee -a /etc/sysctl.conf && sysctl -p
exec streamlit run app.py
