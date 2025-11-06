#!/bin/bash
set -e

echo "=== Docker tøÀ LÜ  ä‰ ==="

# 0t èLt À  p
echo "0t èLt ¬ ..."
docker-compose down 2>/dev/null || true

# tøÀ LÜ
echo "Docker tøÀ LÜ ..."
docker-compose build

# èLt ä‰
echo "èLt Ü‘ ..."
docker-compose up -d

# \ø Ux
echo ""
echo "=== èLt Ü‘( ==="
echo "\ø Ux: docker-compose logs -f"
echo "ÁÜ Ux: docker-compose ps"
echo "À: docker-compose down"
echo ""

# èLt \ø äÜ œ%
docker-compose logs -f
