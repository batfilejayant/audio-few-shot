def mixed_training(
    maml_model,
    bert_model,
    task_sampler,
    bert_dataloader,
    maml_optimizer,
    bert_optimizer,
    epochs=20,
):
    for epoch in range(epochs):
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: MAML Training")
            maml_training(maml_model, task_sampler, maml_optimizer)
        else:
            print(f"Epoch {epoch}: BERT Fine-Tuning")
            bert_training(bert_model, bert_dataloader, bert_optimizer)
