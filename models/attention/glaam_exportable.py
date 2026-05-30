"""
GLAAM Exportable - Attention-exporting version of GLAAM
Extends base GLAAM to optionally return attention weights for XAI visualization
"""

import torch
import torch.nn as nn
from models.attention.glaam import GLAAM, GlobalAttentionBranch, LocalAttentionBranch


class GLAAMExportable(GLAAM):
    """
    Extends base GLAAM to optionally return attention weights for visualization
    Backward compatible with original GLAAM
    """
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tensor (B, C, H, W)
            return_attention: If True, returns (output, attn_dict)
        
        Returns:
            output: Refined features (B, C, H, W)
            attn_dict (optional): {'global': (B, C, 1, 1), 'local': (B, C, H, W)}
        """
        # Get attention weights from both branches
        global_weights = self.global_branch(x)  # (B, C, 1, 1)
        local_weights = self.local_branch(x)    # (B, C, H, W)
        
        # Aggregate: element-wise multiplication
        out = x * global_weights * local_weights
        
        if return_attention:
            attn_dict = {
                'global_weights': global_weights,
                'local_weights': local_weights,
                'combined_attention': global_weights * local_weights  # For visualization
            }
            return out, attn_dict
        
        return out


class GLAAMBlockExportable(nn.Module):
    """
    GLAAM Block with residual connection and attention export
    """
    
    def __init__(self, in_channels: int, reduction: int = 16, use_residual: bool = True):
        super().__init__()
        self.glaam = GLAAMExportable(in_channels, reduction)
        self.use_residual = use_residual
    
    def forward(self, x, return_attention=False):
        if return_attention:
            glaam_out, attn_dict = self.glaam(x, return_attention=True)
            if self.use_residual:
                return x + glaam_out, attn_dict
            return glaam_out, attn_dict
        else:
            glaam_out = self.glaam(x)
            if self.use_residual:
                return x + glaam_out
            return glaam_out


# Test backward compatibility
if __name__ == "__main__":
    print("Testing GLAAM Exportable...")
    
    x = torch.randn(2, 96, 28, 28)
    
    # Original GLAAM
    from models.attention.glaam import GLAAM
    glaam_old = GLAAM(96)
    out_old = glaam_old(x)
    
    # Exportable version
    glaam_new = GLAAMExportable(96)
    out_new = glaam_new(x)
    
    print(f"✅ Backward compatible: {torch.allclose(out_old, out_new, atol=1e-5)}")
    
    # Test attention export
    out_new, attn = glaam_new(x, return_attention=True)
    print(f"✅ Attention export: global={attn['global_weights'].shape}, local={attn['local_weights'].shape}")
    print(f"✅ Combined attention: {attn['combined_attention'].shape}")
