
import matplotlib.pyplot as plt
import numpy as np


def video_frame(sigmoid_vals, output, whole_image, image_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(24, 7)
    ax1.axis('off')
    
    # Plot the original image
    ax1.imshow(np.squeeze(whole_image), alpha=1.0) # rows, columns, channels

    # Plot the original image
    ax2.imshow(np.squeeze(whole_image), alpha=1.0) # rows, columns, channels

    scale = np.array([np.shape(whole_image)[1],np.shape(whole_image)[0]])

    # sigmoid_vals = sigmoid_vals.detach().cpu().numpy()

    # print(np.shape(sigmoid_vals))

    sigmoid_vals = np.squeeze(sigmoid_vals)
    if np.shape(sigmoid_vals)[0] == 45:
        sigmoid_vals = np.reshape(sigmoid_vals, (5, 9)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 32:
        sigmoid_vals = np.reshape(sigmoid_vals, (4, 8)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 35:
        sigmoid_vals = np.reshape(sigmoid_vals, (5, 7)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 40:
        sigmoid_vals = np.reshape(sigmoid_vals, (5, 8)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 54:
        sigmoid_vals = np.reshape(sigmoid_vals, (6, 9)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 77:
        sigmoid_vals = np.reshape(sigmoid_vals, (7, 11)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 6:
        sigmoid_vals = np.reshape(sigmoid_vals, (2, 3)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 28:
        sigmoid_vals = np.reshape(sigmoid_vals, (4, 7)) # rows, columns
    else:
        print("Not sure how many rows and columns!", np.shape(sigmoid_vals))

    CMAP = [[255,20,147],[4, 179, 12],[255,165,0],[131,69,63]]
    CMAP = np.asarray(CMAP)
    colour_predictions = CMAP[sigmoid_vals]

    # Plot the heatmap as a transparent overlay
    offs = np.array([scale[0]/sigmoid_vals.shape[1], scale[1]/sigmoid_vals.shape[0]])

    # Add the sigmoid values for each patch as a label
    for pos, val in np.ndenumerate(sigmoid_vals):
        ax2.annotate(val, xy=np.array(pos)[::-1]*offs+offs/2, ha="center", va="center", fontsize=30)

    heatmap = ax2.imshow(np.flipud(colour_predictions), alpha=0.5, aspect="auto", extent=(0,scale[0],0,scale[1])) # alpha -> more transparent as value decreases

    ax2.invert_yaxis()
    ax2.axis('off')

    if output == 1:
        ax2.set_title('DEPLOY')
    else:
        ax2.set_title('NO DEPLOY')

    # Save the final figure as a .png
    plt.tight_layout()
    plt.savefig(image_name+".png", bbox_inches='tight', pad_inches = 0)
    plt.close(fig)