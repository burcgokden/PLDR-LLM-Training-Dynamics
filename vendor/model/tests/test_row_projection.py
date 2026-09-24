"""The retained-index row maps preserve their declared means and compose."""
import torch
from model_rg.row_projection import project_rows, project_with_exceptions


def test_nested_exception_maps_and_covariance_deficit():
    generator=torch.Generator().manual_seed(107)
    x=torch.randn(3,2,11,7,generator=generator,dtype=torch.float64)
    projected,large=project_with_exceptions(x,4);small=large[...,:1]
    coarse=project_rows(x,small)
    torch.testing.assert_close(projected.mean(-2),x.mean(-2),rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(project_rows(projected,small),coarse,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(project_rows(coarse,large),coarse,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(project_rows(projected,large),projected,rtol=1e-13,atol=1e-13)
    centered=x-x.mean(-2,keepdim=True);kept=projected-x.mean(-2,keepdim=True);error=x-projected
    original_cov=centered.transpose(-1,-2)@centered/11
    reduced_cov=kept.transpose(-1,-2)@kept/11
    error_cov=error.transpose(-1,-2)@error/11
    torch.testing.assert_close(original_cov,reduced_cov+error_cov,rtol=1e-13,atol=1e-13)
    assert torch.linalg.eigvalsh(original_cov-reduced_cov).min()>-1e-12


def test_zero_and_full_retention_preserve_their_defined_limits():
    x=torch.arange(60,dtype=torch.float64).reshape(3,4,5)
    zero,_=project_with_exceptions(x,0);full,_=project_with_exceptions(x,4)
    torch.testing.assert_close(zero,x.mean(-2,keepdim=True).expand_as(x),rtol=0,atol=0)
    torch.testing.assert_close(full,x,rtol=0,atol=0)
