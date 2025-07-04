import numpy as np
import torch
import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

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
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            epoch + 1, num_epochs+1, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    # Save trained model + likelihood
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'train_x': train_x,
        'train_y': train_y
    }, "gp_policy_model.pt")
    print("Model saved as gp_policy_model.pt")

    baseline = torch.tensor([0.0, 0.0, 0.0, 0.0])
    theta_vals = torch.linspace(-math.pi, math.pi, 100)
    test_x = baseline.repeat(100, 1)
    test_x[:, 2] = theta_vals

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    # Extract predictions
    mean = observed_pred.mean.numpy()
    lower, upper = observed_pred.confidence_region()
    lower = lower.numpy()
    upper = upper.numpy()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(theta_vals.numpy(), mean, label='Mean prediction')
    plt.fill_between(theta_vals.numpy(), lower, upper, alpha=0.3, label='Confidence')
    plt.xlabel("Pole angle θ (radians)")
    plt.ylabel("Predicted value (e.g., action)")
    plt.title("GP prediction vs. pole angle (others fixed)")
    plt.legend()
    plt.grid(True)
    plt.savefig("gp_prediction_plot.png")