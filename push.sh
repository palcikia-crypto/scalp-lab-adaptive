#!/data/data/com.termux/files/usr/bin/sh
git add .
git commit -m "Автообновление с телефона: $(date +"%Y-%m-%d %H:%M")"
git push origin master
termux-toast "Успешно обновлено на GitHub!"
