from ceo_chatbot.rag.pipeline import RagService


def main() -> None:
    rag = RagService()

    question = "how to create a pipeline object?"
    answer, docs = rag.answer(question)

    print("==================================Answer==================================")
    print(answer)
    print("==================================Source docs==================================")
    for i, doc in enumerate(docs):
        print(f"Document {i}------------------------------------------------------------")
        print(doc.page_content)


if __name__ == "__main__":
    main()
