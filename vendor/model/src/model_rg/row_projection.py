"""Mean-preserving row projections with explicitly retained exception indices."""
import torch


def project_rows(matrix, indices):
    """Keep selected rows and replace their complement by its own centroid."""
    rows=matrix.shape[-2];count=indices.shape[-1]
    if indices.shape[:-1]!=matrix.shape[:-2] or indices.dtype!=torch.long:
        raise ValueError('Exception indices must match the leading matrix dimensions')
    if count>rows or (count and (indices.min()<0 or indices.max()>=rows)):
        raise ValueError('Invalid exception index')
    if count and torch.any(indices.sort(-1).values.diff(dim=-1)==0):
        raise ValueError('Exception indices must be distinct')
    if count==rows:return matrix.clone()
    mask=torch.ones(matrix.shape[:-1],dtype=torch.bool,device=matrix.device)
    mask.scatter_(-1,indices,False)
    centroid=(matrix*mask.unsqueeze(-1)).sum(-2,keepdim=True)/(rows-count)
    return torch.where(mask.unsqueeze(-1),centroid,matrix)


def project_with_exceptions(matrix, count):
    """Select large row deviations once; the returned indices are retained state."""
    if not 0<=count<=matrix.shape[-2]:raise ValueError('Invalid exception count')
    mean=matrix.mean(-2,keepdim=True)
    distances=(matrix-mean).square().sum(-1)
    indices=distances.argsort(dim=-1,descending=True,stable=True)[...,:count]
    return project_rows(matrix,indices),indices
