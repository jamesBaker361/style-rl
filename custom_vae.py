from diffusers import AutoencoderKL
import torch
import functools

def public_encode(vae:AutoencoderKL,x: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = x.shape

    if vae.use_tiling and (width > vae.tile_sample_min_size or height > vae.tile_sample_min_size):
        return public_tiled_encode(vae,x)

    enc = vae.encoder(x)
    if vae.quant_conv is not None:
        enc = vae.quant_conv(enc)

    return enc

def public_tiled_encode(vae:AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.Tensor`): Input batch of images.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """

        overlap_size = int(vae.tile_sample_min_size * (1 - vae.tile_overlap_factor))
        blend_extent = int(vae.tile_latent_min_size * vae.tile_overlap_factor)
        row_limit = vae.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + vae.tile_sample_min_size, j : j + vae.tile_sample_min_size]
                tile = vae.encoder(tile)
                if vae.config.use_quant_conv:
                    tile = vae.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = vae.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = vae.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        enc = torch.cat(result_rows, dim=2)
        return enc
