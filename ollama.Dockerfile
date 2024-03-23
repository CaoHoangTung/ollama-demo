FROM ollama/ollama:0.1.9

RUN ollama pull mistral:instruct

ENTRYPOINT ["/bin/ollama"]

CMD ["serve"]