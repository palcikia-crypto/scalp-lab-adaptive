#!/data/data/com.termux/files/usr/bin/sh
echo "Стягиваем изменения с GitHub..."
git pull origin master --no-rebase
echo "Отправляем локальные изменения..."
git add .
git commit -m "Синхронизация с телефона: $(date +"%Y-%m-%d %H:%M")"
git push origin master
termux-toast "Синхронизация завершена!"
