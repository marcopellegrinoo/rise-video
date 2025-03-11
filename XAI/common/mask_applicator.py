import copy

def multiplicative_uniform_noise_onechannel_spatial(images, masks, channel, **kwargs):
    std_zero_value = kwargs.get("std_zero_value", -0.6486319166678826)

    masked = []

    # Itero su tutte le N maschere generate
    for mask in masks:
        masked_images = copy.deepcopy(images)  # Copia profonda delle immagini originali

        # Applica la perturbazione solo al canale specificato
        masked_images[..., channel] = (
            masked_images[..., channel] * mask + (1 - mask) * std_zero_value)

        masked.append(masked_images)

    return masked