import cv2
import os

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to capture images
def capture_images(name, image_count):
    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Create a directory with the person's name
    create_directory(name)

    # Counter for image filenames
    count = 0

    # Loop to capture images
    while count < image_count:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow('Capture Images', frame)

        # Prompt the user to press 'Enter' to capture an image
        if cv2.waitKey(1) & 0xFF == 13:  # 13 is the Enter Key
            # Save the image with the person's name
            image_path = os.path.join(name, f'{name}_{count}.jpg')
            cv2.imwrite(image_path, frame)
            print(f'Image {count + 1} saved as {image_path}')
            count += 1

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Prompt the user to enter their name
    name = input("Enter your name: ")

    # Number of images to capture
    image_count = 80

    # Capture images
    capture_images(name, image_count)

if __name__ == "__main__":
    main()
