from src import check_file, textsplitter


if __name__ == "__main__":
    skills = check_file.check_skill()
    splitter = textsplitter.RecursiveCharacterTextSplitter(
        chunk_size= 256,
        chunk_overlap= 32,
        length_function= len,
        separators= ['\n\n', '\n', '']
    )
    for skill in skills:
        with open("result.txt", "a", encoding="utf-8") as f:
            f.write(f"Skill: {skill}\n{'='*40}\n")
        file_paths = check_file.check_single_skill(skill)
        content = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                content.append(f.read())
        chunks = splitter.create_documents(content)
        with open("result.txt", "a", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"Chunk {i} (length {len(chunk)}):\n{chunk}\n{'-'*40}\n")