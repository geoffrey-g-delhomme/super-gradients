import torch

def image2border(x, y, w, h, borders):
    z = 0.
    borders += borders[0]
    imax = len(borders) - 1
    for i, border in enumerate(borders):
        if border == 't':
            if i == 0:
                if y == 0 and x >= w/2:
                    z += (x - w/2)
                    break
                else:
                    z += w/2
            elif i == imax:
                z += x
            else:
                if y == 0:
                    z += x
                    break
                else:
                    z += w
        elif border == 'r':
            if i == 0:
                if x == w and y >= h/2:
                    z += (y - h/2)
                    break
                else:
                    z += h/2
            elif i == imax:
                z += y
            else:
                if x == w:
                    z += y
                    break
                else:
                    z += h
        elif border == 'b':
            if i == 0:
                if y == h and (w - x) >= w/2:
                    z += (w - x) - w/2
                    break
                else:
                    z += w/2
            elif i == imax:
                z += (w - x)
            else:
                if y == h:
                    z += (w - x)
                    break
                else:
                    z += w
        elif border == 'l':
            if i == 0:
                if x == 0 and (h - y) >= h/2:
                    z += (h - y) - h/2
                    break
                else:
                    z += h/2
            elif i == imax:
                z += (h - y)
            else:
                if x == 0:
                    z += (h - y)
                    break
                else:
                    z += h
    return z

def border2image(z, w, h, borders):
    if borders == "trbl":
        x = (
            torch.clip(z, 0., w/2) + w/2 +
            torch.clip(w - (z - w/2 - h), 0., w) - w +
            torch.clip(z - w/2 - h - w - h, 0., w/2)
        )
        y = (
            torch.clip(z - w/2, 0., h) +
            torch.clip(h - (z - w/2 - h - w), 0., h) - h
        )
    elif borders == "rblt":
        x = (
            torch.clip(w - (z - h/2), 0., w) - w +
            torch.clip(z - h/2 - w - h, 0., w) + w
        )
        y = (
            torch.clip(z, 0., h/2) + h/2 +
            torch.clip(h - (z - h/2 - w), 0., h) - h +
            torch.clip(z - h/2 - w - h - w, 0., h/2)
        )
    elif borders == "bltr":
        x = (
            torch.clip(w/2 - z, 0., w/2) +
            torch.clip(z - w/2 - h, 0., w) +
            torch.clip(w/2 - (z - w/2 - h - w - h), 0., w/2) - w/2
        )
        y = (
            torch.clip(h - (z - w/2), 0., h) +
            torch.clip(z - w/2 - h - w, 0., h)
        )
    elif borders == "ltrb":
        x = (
            torch.clip(z - h/2, 0., w) +
            torch.clip(w - (z - h/2 - w - h), 0., w) - w
        )
        y = (
            torch.clip(h/2 - z, 0., h/2) +
            torch.clip(z - h/2 - w, 0., h) +
            torch.clip(h/2 - (z - h/2 - w - h - w), 0., h/2) - h/2
        )
    return torch.stack((x, y), dim=-1)