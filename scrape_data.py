import requests
from bs4 import BeautifulSoup
import hashlib
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0"
}

BASE_URL = "https://vienthammydiva.vn/"

def hash_text(text: str) -> str:
    """Tạo hash SHA-256 từ một đoạn văn."""
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_article_links(page_url):
    """Lấy tất cả link bài viết từ trang chủ hoặc chuyên mục."""
    try:
        res = requests.get(page_url, headers=HEADERS)
        soup = BeautifulSoup(res.text, "lxml")

        a_tags = soup.find_all("a")
        links = set()

        for a in a_tags:
            href = a.get("href")
            if href and href.startswith(BASE_URL):
                if href != BASE_URL:
                    links.add(href)

        return links
    except Exception as e:
        print(f"❌ Lỗi khi lấy links từ {page_url}: {e}")
        return set()

def get_article_content(url):
    """Lấy toàn bộ nội dung từ các thẻ h1-h6 và p, loại bỏ trùng lặp đoạn bằng hash."""
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "lxml")

        tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])

        seen_hashes = set()
        texts = []

        for tag in tags:
            text = tag.get_text(strip=True)
            if text and len(text) > 20:  # lọc đoạn quá ngắn
                h = hash_text(text)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    texts.append(text)

        return "\n\n".join(texts)  # ngắt đoạn rõ ràng
    except Exception as e:
        print(f"❌ Lỗi khi đọc nội dung từ {url}: {e}")
        return ""

def main():
    os.makedirs('data', exist_ok=True)
    article_links = get_article_links(BASE_URL)
    print(f"Tìm thấy {len(article_links)} link bài viết.")

    for i, link in enumerate(article_links, 1):
        print(f"\nĐang xử lý bài viết {i}/{len(article_links)}: {link}")
        content = get_article_content(link)
        if content:
            file_path = os.path.join('data', f'article_{i}.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Đã lưu bài viết vào {file_path}")
        else:
            print("Không lấy được nội dung.")

if __name__ == "__main__" :
    main()
