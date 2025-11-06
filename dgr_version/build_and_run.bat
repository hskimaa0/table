@echo off
chcp 65001 >nul
echo === Docker tøÀ LÜ  ä‰ ===
echo.

REM 0t èLt À  p
echo 0t èLt ¬ ...
docker-compose down 2>nul

REM tøÀ LÜ
echo Docker tøÀ LÜ ...
docker-compose build

REM èLt ä‰
echo èLt Ü‘ ...
docker-compose up -d

echo.
echo === èLt Ü‘( ===
echo \ø Ux: docker-compose logs -f
echo ÁÜ Ux: docker-compose ps
echo À: docker-compose down
echo.

REM èLt \ø äÜ œ%
docker-compose logs -f
