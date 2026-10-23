import fitz


class ResumeParser:

    @staticmethod
    def parse(file_path: str) -> str:

        document = fitz.open(file_path)

        pages = []

        for page in document:

            text = page.get_text("text")

            if text.strip():
                pages.append(text)

        document.close()

        return "\n".join(pages)