import re


class ResumeSectionService:

    SECTION_ALIASES = {
        "summary": {
            "summary",
            "profile",
            "profile summary",
            "professional summary",
            "professional profile",
            "executive summary",
            "career summary",
            "career profile",
            "about",
            "about me",
            "objective",
            "career objective",
            "professional objective",
        },
        "skills": {
            "skills",
            "technical skills",
            "core skills",
            "key skills",
            "technical expertise",
            "technical competencies",
            "core competencies",
            "areas of expertise",
            "professional skills",
            "competencies",
            "technologies",
            "technology stack",
            "tech stack",
        },
        "work_experience": {
            "experience",
            "work experience",
            "professional experience",
            "employment history",
            "work history",
            "career history",
            "professional background",
            "employment experience",
            "relevant experience",
            "internship",
            "internships",
        },
        "projects": {
            "projects",
            "project",
            "project experience",
            "personal projects",
            "academic projects",
            "key projects",
            "major projects",
            "selected projects",
            "professional projects",
        },
    }

    IGNORED_SECTIONS = {
        "contact",
        "contact information",
        "personal information",
        "education",
        "educational qualification",
        "educational qualifications",
        "academic background",
        "academic qualifications",
        "education and qualifications",
        "training and education",
        "certifications",
        "certificates",
        "achievements",
        "achievement",
        "achivement",
        "awards",
        "languages",
        "references",
        "interests",
        "hobbies",
        "declaration",
        "personal details",
        "other personal details",
        "additional information",
        "volunteer experience",
        "publications",
    }

    DATE_RE = re.compile(
        r"\b((jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s*)?"
        r"(19|20)\d{2}\b|\bpresent\b",
        re.I,
    )
    EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
    PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")

    @classmethod
    def extract_relevant_content(cls, text: str) -> dict:
        lines = cls._prepare_lines(text)
        sections = cls._split_sections(lines)
        output = {
            "summary": cls._extract_summary(sections),
            "skills": cls._extract_skills(sections),
            "projects": cls._extract_projects(sections),
            "work_experience": cls._extract_work_experience(sections),
        }

        return {"summary": output.get("summary", ""),"skills": output.get("skills", []),"projects": output.get("projects", []),"work_experience": output.get("work_experience", [])
}
    @classmethod
    def extract_sections(cls, text: str) -> dict:
        content = cls.extract_relevant_content(text)

        return {
            key: cls._section_to_text(value)
            for key, value in content.items()
            if value
        }

    @classmethod
    def _prepare_lines(cls, text: str) -> list[str]:
        return [
            line
            for line in (cls._clean_line(raw_line) for raw_line in text.splitlines())
            if line
        ]

    @classmethod
    def _split_sections(cls, lines: list[str]) -> dict:
        sections = {
            "summary": [],
            "skills": [],
            "projects": [],
            "work_experience": [],
            "ignored": [],
            "unassigned": [],
        }
        current = "unassigned"
        seen_relevant_heading = False

        for line in lines:
            detected, inline_content = cls._detect_heading(line)

            if detected:
                current = detected
                if detected != "ignored":
                    seen_relevant_heading = True
                if inline_content:
                    sections[current].append(inline_content)
                continue

            if cls._is_probable_heading(line):
                current = "unassigned"
                continue

            if current == "ignored" and cls._looks_like_work_period(line):
                sections["work_experience"].append(line)
                continue

            if current == "ignored":
                sections["ignored"].append(line)
                continue

            if (
                current == "unassigned"
                and not seen_relevant_heading
                and cls._looks_like_front_matter_skill(line)
            ):
                sections["skills"].append(line)
                continue

            sections[current].append(line)

        return sections

    @classmethod
    def _recover_column_order(cls, sections: dict) -> dict:
        return sections

    @classmethod
    def _extract_summary(cls, sections: dict) -> str:
        lines = sections["summary"]

        if not lines:
            lines = cls._summary_candidates(sections["unassigned"])

        summary = []

        for line in lines:
            if cls._looks_like_project_title(line) or cls._looks_like_date_line(line):
                break
            summary.append(line)

        return cls._join_text(summary)

    @classmethod
    def _extract_skills(cls, sections: dict) -> list[str]:
        skills = []

        skill_lines = sections["skills"][:]

        if not skill_lines:
            skill_lines = cls._front_matter_skills(
                sections["ignored"] + sections["unassigned"]
            )

        for line in skill_lines:
            if cls._is_contact_noise(line):
                continue

            parts = re.split(r"[,|;/•]+|\s{2,}", line)
            skills.extend(part.strip(" -:\t") for part in parts if part.strip(" -:\t"))

        return cls._unique(skills)

    @classmethod
    def _extract_projects(cls, sections: dict) -> list[dict]:
        lines = [
            line
            for line in sections["projects"]
            if not cls._looks_like_front_matter_skill(line)
        ]

        if not lines:
            lines = cls._project_candidate_lines(
                sections["summary"] + sections["unassigned"]
            )

        projects = []
        current_index = None

        pending_descriptions = []

        for line in lines:
            if cls._is_project_noise(line):
                continue

            if cls._looks_like_project_title(line):
                projects.append({
                    "name": cls._clean_name(line),
                    "description": "",
                })
                current_index = len(projects) - 1
                continue

            pending_descriptions.append(line)

        if pending_descriptions and projects:
            cls._assign_project_descriptions(projects, pending_descriptions)

        return [
                {
                    "name": project.get("name", ""),
                    "description": project.get("description", ""),
                    "raw_text": cls._append_text(
                        project.get("name", ""),
                        project.get("description", "")
                    )
                }
                for project in projects
                if project.get("name")
            ]

    @classmethod
    def _assign_project_descriptions(cls, projects: list[dict], descriptions: list[str]) -> None:
        current_index = 0

        for line in descriptions:
            if not projects or cls._is_project_noise(line):
                continue

            first_empty_index = cls._first_empty_description_index(projects)

            if (
                current_index is None
                or projects[current_index]["description"].endswith(".")
                or (
                    not projects[current_index]["description"]
                    and first_empty_index < current_index
                )
            ):
                current_index = first_empty_index

            projects[current_index]["description"] = cls._append_text(
                projects[current_index]["description"],
                line,
            )

    @classmethod
    def _extract_work_experience(cls, sections: dict) -> list[dict]:

        lines = [
            line
            for line in sections["work_experience"]
            if not cls._is_contact_noise(line)
        ]

        items = []
        current_item = None

        for line in lines:

            if cls._looks_like_work_period(line):

                if current_item:
                    items.append(current_item)

                current_item = {
                    "company": "",
                    "role": line,
                    "description": "",
                    "raw_text": line
                }

                continue

            if current_item is None:
                current_item = {
                    "company": "",
                    "role": "",
                    "description": "",
                    "raw_text": ""
                }

            if not current_item["company"]:
                current_item["company"] = line

            elif not current_item["description"]:
                current_item["description"] = line

            else:
                current_item["description"] = cls._append_text(
                    current_item["description"],
                    line
                )

            current_item["raw_text"] = cls._append_text(
                current_item["raw_text"],
                line
            )

        if current_item:
            items.append(current_item)

        return items

    @classmethod
    def _summary_candidates(cls, lines: list[str]) -> list[str]:
        return [
            line
            for line in lines
            if not cls._is_contact_noise(line)
            and not cls._looks_like_front_matter_skill(line)
            and not cls._looks_like_short_identity_line(line)
        ]

    @classmethod
    def _front_matter_skills(cls, lines: list[str]) -> list[str]:
        skills = []

        for line in lines:
            if cls._looks_like_project_title(line) or cls._looks_like_date_line(line):
                break

            if cls._looks_like_front_matter_skill(line):
                skills.append(line)

        return skills

    @classmethod
    def _project_candidate_lines(cls, lines: list[str]) -> list[str]:
        for index, line in enumerate(lines):
            if cls._looks_like_project_title(line):
                return lines[index:]

        return []

    @staticmethod
    def _clean_line(line: str) -> str:
        line = line.replace("\ufffd", "-")
        line = re.sub(r"\s+", " ", line)
        return line.strip(" •\t")

    @staticmethod
    def _normalize_heading(line: str) -> str:
        line = line.lower().strip()
        line = re.sub(r"^\d{1,2}\s*[.)|:\-]\s*", "", line)
        line = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", line)
        line = re.sub(r"[:\-]+$", "", line)
        line = re.sub(r"\s+", " ", line)
        return line.strip()

    @classmethod
    def _find_section(cls, heading: str) -> str | None:
        for section, aliases in cls.SECTION_ALIASES.items():
            if heading in aliases:
                return section

        return None

    @classmethod
    def _detect_heading(cls, line: str) -> tuple[str | None, str]:
        """Return a canonical section name and any content after an inline heading.

        Examples handled here include ``SKILLS:``, ``01 | EXPERIENCE`` and
        ``Technical Skills: Python, SQL``.  Matching remains alias-based so a
        normal sentence containing the word "experience" is not a boundary.
        """
        heading = cls._normalize_heading(line)
        detected = cls._find_section(heading)

        if detected:
            return detected, ""
        if heading in cls.IGNORED_SECTIONS:
            return "ignored", ""

        for separator in (":", "|"):
            if separator not in line:
                continue
            prefix, content = line.split(separator, 1)
            normalized_prefix = cls._normalize_heading(prefix)
            detected = cls._find_section(normalized_prefix)
            if detected:
                return detected, cls._clean_line(content)
            if normalized_prefix in cls.IGNORED_SECTIONS:
                return "ignored", cls._clean_line(content)

        return None, ""

    @classmethod
    def _is_probable_heading(cls, line: str) -> bool:
        heading = cls._normalize_heading(line)

        return (
            bool(heading)
            and len(line) <= 45
            and 2 <= len(heading.split()) <= 5
            and not line.endswith(".")
            and line.isupper()
            and not cls._is_contact_noise(line)
            and not cls._looks_like_date_line(line)
            and not cls._looks_like_front_matter_skill(line)
        )

    @classmethod
    def _looks_like_project_title(cls, line: str) -> bool:
        clean = line.strip(" -:")

        if len(clean) < 4 or len(clean) > 90:
            return False

        if cls._is_contact_noise(clean) or cls._looks_like_date_line(clean):
            return False

        if re.search(r"[,;|]", clean):
            return False

        if re.search(r"\b(school|college|university|polytechnic|govt\.?)\b", clean, re.I):
            return False

        if re.search(r"\b(awarded|employee|month|certificate|certification)\b", clean, re.I):
            return False

        if re.search(r"\([^)]{2,40}\)", clean):
            return True

        if re.search(r"\s[-–—]\s", clean):
            return True

        if re.search(r"\b(award|awarded|achievement|achivement|certificate|certification)\b", clean, re.I):
            return False

        words = clean.split()
        titled_words = sum(1 for word in words if word[:1].isupper())

        return 3 <= len(words) <= 7 and titled_words >= max(2, len(words) - 1)

    @classmethod
    def _looks_like_front_matter_skill(cls, line: str) -> bool:
        if cls._is_contact_noise(line) or cls._looks_like_date_line(line):
            return False

        if re.search(r"\d", line):
            return False

        first_word = line.split()[0] if line.split() else ""

        if re.search(r"(ed|ing)$", first_word, re.I) or line.lower().endswith(" and"):
            return False

        if re.search(r"[,|;/•]", line):
            return True

        words = line.split()

        if not 1 <= len(words) <= 4:
            return False

        if any(char in line for char in ("#", ".", "+")):
            return True

        if line.isupper():
            return all(len(word.strip("-")) <= 5 for word in words)

        return (
            all(word[:1].isupper() for word in words)
            and any(sum(1 for char in word if char.isupper()) >= 2 for word in words)
        )

    @classmethod
    def _looks_like_short_identity_line(cls, line: str) -> bool:
        words = line.split()

        return 1 <= len(words) <= 4 and line.isupper()

    @classmethod
    def _looks_like_date_line(cls, line: str) -> bool:
        return bool(cls.DATE_RE.search(line))

    @classmethod
    def _looks_like_work_period(cls, line: str) -> bool:
        return bool(
            cls.DATE_RE.search(line)
            and (
                re.search(r"[-–—]", line)
                or re.search(r"\bpresent\b", line, re.I)
            )
        )

    @classmethod
    def _is_project_noise(cls, line: str) -> bool:
        clean = line.strip()

        return bool(
            cls._is_contact_noise(clean)
            or cls._looks_like_work_period(clean)
            or (cls._looks_like_front_matter_skill(clean) and len(clean.split()) <= 4)
            or (len(clean.split()) == 1 and clean[:1].isupper())
            or re.fullmatch(r"\d+(st|nd|rd|th)?", clean, re.I)
            or re.search(r"\b(school|college|university|polytechnic|institute)\b", clean, re.I)
            or re.search(r"\b(award|awarded|achievement|achivement|certificate|certification)\b", clean, re.I)
        )

    @classmethod
    def _is_contact_noise(cls, line: str) -> bool:
        return bool(cls.EMAIL_RE.search(line) or cls.PHONE_RE.search(line))

    @staticmethod
    def _clean_name(line: str) -> str:
        return re.sub(r"\s+", " ", line).strip(" -:")

    @staticmethod
    def _join_text(lines: list[str]) -> str:
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    @staticmethod
    def _append_text(existing: str, line: str) -> str:
        return f"{existing} {line}".strip() if existing else line

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        unique_items = []
        seen = set()

        for item in items:
            key = item.lower()

            if key not in seen:
                unique_items.append(item)
                seen.add(key)

        return unique_items

    @staticmethod
    def _first_empty_description_index(projects: list[dict]) -> int:
        for index, project in enumerate(projects):
            if not project["description"]:
                return index

        return len(projects) - 1

    @staticmethod
    def _section_to_text(value) -> str:
        if isinstance(value, str):
            return value

        if isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                return "\n".join(value)

            return "\n".join(
                " ".join(str(part) for part in item.values() if part)
                for item in value
            )

        return str(value)
