class Task:
    name = ""
    higher_is_better = True
    max_steps_cap = 1000
    checkpoint_every = 50

    def build_model(self, seed):
        raise NotImplementedError

    def build_data(self, seed):
        raise NotImplementedError

    def sample_batch(self, state):
        loader, iterator = state
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch, (loader, iterator)

    def loss_and_metric(self, model, batch):
        raise NotImplementedError

    def evaluate(self, model, val_loader):
        raise NotImplementedError

    def run(self, optimizer_cls, kwargs, step_budget, seed):
        train_loader, val_loader = self.build_data(seed)
        model = self.build_model(seed)
        optimizer = optimizer_cls(model.parameters(), **kwargs)
        history = []
        state = (train_loader, iter(train_loader)) if hasattr(train_loader, "__iter__") else train_loader
        step = 0
        while step < step_budget:
            batch, state = self.sample_batch(state)
            optimizer.zero_grad()
            loss = self.loss_and_metric(model, batch)
            loss.backward()
            optimizer.step()
            step += 1
            if step % self.checkpoint_every == 0 or step == step_budget:
                metric = self.evaluate(model, val_loader)
                history.append((step, metric))
        return history
