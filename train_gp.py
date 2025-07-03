import numpy as np
import torch
import gpytorch

# 1) Multiclass GP model for classification
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        #print(f"Mean shape: {mean_x.shape}")
        #print(f"Covariance shape: {covar_x.shape}")

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def preprocess(path="gp_training_upright.npz"):
    data = np.load(path)
    states = data["states"]
    actions = data["actions"].ravel()

    print(f"Number of actions: {len(actions)}")
    print(f"Raw states shape: {states.shape}")
    print(f"Raw actions shape: {actions.shape}")

    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
    )


if __name__ == "__main__":
    train_x, train_y = preprocess()

    print(f"Train x shape: {train_x.shape}")
    print(f"Train y shape: {train_y.shape}")

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()}
    ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    num_epochs = 200
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} — Loss: {loss.item():.4f}")

    # Save trained model + likelihood
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
    }, "gp_policy_model.pt")
