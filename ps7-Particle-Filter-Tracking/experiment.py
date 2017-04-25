"""Problem Set 7: Particle Filter Tracking"""

from ps7 import *

# I/O directories
input_dir = "input"
output_dir = "output"

# Driver/helper code
def run_particle_filter1(pf_class, video_filename, template_rect, save_frames={}, save_video=False, **kwargs):
    """Instantiates and runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame, template (extracted from first frame using
    template_rect), and any keyword arguments.

    Do not modify this function except for the debugging flag used to display every frame.

    Args:
        pf_class (object): particle filter class to instantiate (e.g. ParticleFilter).
        video_filename (str): path to input video file.
        template_rect (dict): template bounds (x, y, w, h), as float or int.
        save_frames (dict): frames to save {<frame number>|'template': <filename>}.
        save_video (bool): save output video
        **kwargs: arbitrary keyword arguments passed on to particle filter class.

    Returns:
        None.

    """
    watermark_name = "Nikhil G"
    out_video_name = 'test4.mp4'

    # Open video file
    video = cv2.VideoCapture(video_filename)
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(out_video_name, -1, 20.0, (1280, 720))

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (until last frame or Ctrl + C is pressed)
    while True:
        try:
            # Try to read a frame
            okay, frame = video.read()
            if not okay:
                break  # no more frames, or can't read video

            # Extract template and initialize (one-time only)
            if template is None:
                template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
                                 int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]
                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)
                pf = pf_class(frame, template, **kwargs)

            # Process frame
            pf.process(frame)

            if False:  # For debugging, it displays every frame
                frame_out = frame.copy()
                pf.render(frame_out)
                cv2.imshow('Tracking', frame_out)
                cv2.waitKey(0)

            # Render and save output, if indicated
            frame_out = frame.copy()
            if frame_num in save_frames:
                pf.render(frame_out)
                cv2.imwrite(save_frames[frame_num], frame_out)

            if save_video:
                pf.render(frame_out)
                cv2.putText(img=frame_out,
                            text=watermark_name,
                            org=(215, 350),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=4,
                            color=(200, 200, 200),
                            thickness=2)
                out.write(frame_out)

            # Update frame number
            frame_num += 1

        except KeyboardInterrupt:  # press ^C to quit
            break

# Driver/helper code
def run_particle_filter(pf_class, video_filename, template_rect, save_frames={}, **kwargs):
    """Instantiates and runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame, template (extracted from first frame using
    template_rect), and any keyword arguments.

    Do not modify this function except for the debugging flag used to display every frame.

    Args:
        pf_class (object): particle filter class to instantiate (e.g. ParticleFilter).
        video_filename (str): path to input video file.
        template_rect (dict): template bounds (x, y, w, h), as float or int.
        save_frames (dict): frames to save {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle filter class.

    Returns:
        None.

    """

    # Open video file
    video = cv2.VideoCapture(video_filename)

    # Initialize objects
    template = None
    pf = None
    frame_num = 0
    write_frame0 = False

    # Loop over video (until last frame or Ctrl + C is presssed)
    while True:
        try:
            # Try to read a frame
            okay, frame = video.read()
            if not write_frame0:
                cv2.imwrite(os.path.join(output_dir, 'pedestrians-frame0.png'), frame)
                write_frame0 = True
            if not okay:
                break  # no more frames, or can't read video

            # Extract template and initialize (one-time only)
            if template is None:
                template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
                                 int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]
                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)
                pf = pf_class(frame, template, **kwargs)

            # Process frame
            pf.process(frame)  

            if False:  # For debugging, it displays every frame
                out_frame = frame.copy()
                pf.render(out_frame)
                cv2.imshow('Tracking', out_frame)
                cv2.waitKey(1)

            # Render and save output, if indicated
            if frame_num in save_frames:
                frame_out = frame.copy()
                pf.render(frame_out)
                cv2.imwrite(save_frames[frame_num], frame_out)

            # Update frame number
            frame_num += 1

        except KeyboardInterrupt:  # press ^C to quit
            break


def part_0a():
    num_particles = 0.  # Define the number of particles
    sigma_mse = 0.  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 0.  # Define the value of sigma for the particles movement (dynamics)
    template_rect = {}

    run_particle_filter(ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "test.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-0-a-1.png'),
                            28: os.path.join(output_dir, 'ps7-0-a-2.png'),
                            94: os.path.join(output_dir, 'ps7-0-a-3.png'),
                            171: os.path.join(output_dir, 'ps7-0-a-4.png')
                        },  # frames to save, mapped to filenames, and 'template' if desired
                        num_particles=num_particles, sigma_exp=sigma_mse, sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to

def part_0a():
    num_particles = 100  # Define the number of particles
    sigma_mse = 10.0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.0  # Define the value of sigma for the particles movement (dynamics)
    template_rect = {'x': 60, 'y': 60, 'w': 80, 'h': 80}

    run_particle_filter(ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "test.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-0-a-1.png'),
                            28: os.path.join(output_dir, 'ps7-0-a-2.png'),
                            94: os.path.join(output_dir, 'ps7-0-a-3.png'),
                            171: os.path.join(output_dir, 'ps7-0-a-4.png')
                        },  # frames to save, mapped to filenames, and 'template' if desired
                        num_particles=num_particles, sigma_exp=sigma_mse, sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to

def part_1a():
    num_particles = 100  # Define the number of particles
    sigma_mse = 10.0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.0  # Define the value of sigma for the particles movement (dynamics)
    template_rect = {'x': int(320.8751), 'y': int(175.1776), 'w': int(102.5404), 'h': int(128.0504)}  # suggested template window (dict)

    run_particle_filter(ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "pres_debate.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-1-a-1.png'),
                            28: os.path.join(output_dir, 'ps7-1-a-2.png'),
                            94: os.path.join(output_dir, 'ps7-1-a-3.png'),
                            171: os.path.join(output_dir, 'ps7-1-a-4.png')
                        },  # frames to save, mapped to filenames, and 'template' if desired
                        num_particles=num_particles, sigma_exp=sigma_mse, sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to


def part_1b():
    num_particles = 100  # Define the number of particles
    sigma_mse = 10.0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.0  # Define the value of sigma for the particles movement (dynamics)
    template_rect = {'x': int(320.8751), 'y': int(175.1776), 'w': int(102.5404), 'h': int(128.0504)}

    run_particle_filter(ParticleFilter,
                        os.path.join(input_dir, "noisy_debate.mp4"),
                        template_rect,
                        {
                            14: os.path.join(output_dir, 'ps7-1-b-1.png'),
                            94: os.path.join(output_dir, 'ps7-1-b-2.png'),
                            530: os.path.join(output_dir, 'ps7-1-b-3.png')
                        },
                        num_particles=num_particles, sigma_exp=sigma_mse, sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to


def part_2a():
    num_particles = 150  # Define the number of particles
    sigma_mse = 10.0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 15.0  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.50  # Set a value for alpha
    template_rect = {'x': 530, 'y': 382, 'w': 94, 'h': 110}  # Define the template window values

    run_particle_filter(AppearanceModelPF,  # particle filter model class
                        os.path.join(input_dir, "pres_debate.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-2-a-1.png'),
                            22: os.path.join(output_dir, 'ps7-2-a-2.png'),
                            50: os.path.join(output_dir, 'ps7-2-a-3.png'),
                            160: os.path.join(output_dir, 'ps7-2-a-4.png')
                        },
                        num_particles=num_particles, sigma_exp=sigma_mse, sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_rect)  # Add more if you need to


def part_2b():
    num_particles = 150  # Define the number of particles
    sigma_mse = 10.0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 15.0  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.50  # Set a value for alpha
    template_rect = {'x': 530, 'y': 382, 'w': 94, 'h': 110}  # Define the template window values

    run_particle_filter(AppearanceModelPF,  # particle filter model class
                        os.path.join(input_dir, "noisy_debate.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-2-b-1.png'),
                            22: os.path.join(output_dir, 'ps7-2-b-2.png'),
                            50: os.path.join(output_dir, 'ps7-2-b-3.png'),
                            160: os.path.join(output_dir, 'ps7-2-b-4.png')
                        },
                        num_particles=num_particles, sigma_exp=sigma_mse, sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_rect)  # Add more if you need to


def part_3a():
    num_particles = 100  # Define the number of particles
    sigma_chi = 10.0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.0  # Define the value of sigma for the particles movement (dynamics)
    hist_bins_num = 8
    template_rect = {'x': int(320.8751), 'y': int(175.1776), 'w': int(102.5404), 'h': int(128.0504)}

    run_particle_filter(MeanShiftLitePF,
                        os.path.join(input_dir, "pres_debate.mp4"),
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-3-a-1.png'),
                            28: os.path.join(output_dir, 'ps7-3-a-2.png'),
                            94: os.path.join(output_dir, 'ps7-3-a-3.png'),
                            171: os.path.join(output_dir, 'ps7-3-a-4.png')
                        },
                        num_particles=num_particles, sigma_exp=sigma_chi, sigma_dyn=sigma_dyn,
                        hist_bins_num=hist_bins_num,
                        template_coords=template_rect)  # Add more if you need to


def part_3b():
    num_particles = 100  # Define the number of particles
    sigma_chi = 10.0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.0  # Define the value of sigma for the particles movement (dynamics)
    hist_bins_num = 6  # Define the number of bins
    alpha = 0.50  # Set a value for alpha
    template_rect = {'x': 548, 'y': 412, 'w': 38, 'h': 46}  # Define the template window values

    run_particle_filter(MeanShiftLitePF,
                        os.path.join(input_dir, "pres_debate.mp4"),
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-3-b-1.png'),
                            22: os.path.join(output_dir, 'ps7-3-b-2.png'),
                            50: os.path.join(output_dir, 'ps7-3-b-3.png'),
                            160: os.path.join(output_dir, 'ps7-3-b-4.png')
                        },
                        num_particles=num_particles, sigma_exp=sigma_chi, sigma_dyn=sigma_dyn, alpha=alpha,
                        hist_bins_num=hist_bins_num,
                        template_coords=template_rect)  # Add more if you need to


def part_4():
    """Contains experiments using the code from part 1 and different parameters.

    Please follow the problem set documentation. In order to make grading easier, copy the code from part 1a and modify
    it to run your tests.

    The results you observe in this part should help with your discussion answers.

    Returns:
        None
    """
    return
    #'''
    num_particles = 100  # Define the number of particles
    sigma_mse = 10.0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.0  # Define the value of sigma for the particles movement (dynamics)
    template_rect = {'x': int(260.8751), 'y': int(115.1776), 'w': int(192.5404), 'h': int(218.0504)}  # suggested template window (dict)

    run_particle_filter(ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "pres_debate.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-4-a-1.png'),
                            28: os.path.join(output_dir, 'ps7-4-a-2.png'),
                            94: os.path.join(output_dir, 'ps7-4-a-3.png'),
                            171: os.path.join(output_dir, 'ps7-4-a-4.png')
                        },  # frames to save, mapped to filenames, and 'template' if desired
                        num_particles=num_particles, sigma_exp=sigma_mse, sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to

    template_rect = {'x': int(350.8751), 'y': int(205.1776), 'w': int(72.5404), 'h': int(98.0504)}  # suggested template window (dict)

    run_particle_filter(ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "pres_debate.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-4-b-1.png'),
                            28: os.path.join(output_dir, 'ps7-4-b-2.png'),
                            94: os.path.join(output_dir, 'ps7-4-b-3.png'),
                            171: os.path.join(output_dir, 'ps7-4-b-4.png')
                        },  # frames to save, mapped to filenames, and 'template' if desired
                        num_particles=num_particles, sigma_exp=sigma_mse, sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to


    num_particles = 100  # Define the number of particles
    sigma_mse = 1.0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.0  # Define the value of sigma for the particles movement (dynamics)
    template_rect = {'x': int(320.8751), 'y': int(175.1776), 'w': int(102.5404), 'h': int(128.0504)}  # suggested template window (dict)

    run_particle_filter(ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "pres_debate.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-4-c-1.png'),
                            28: os.path.join(output_dir, 'ps7-4-c-2.png'),
                            94: os.path.join(output_dir, 'ps7-4-c-3.png'),
                            171: os.path.join(output_dir, 'ps7-4-c-4.png')
                        },  # frames to save, mapped to filenames, and 'template' if desired
                        num_particles=num_particles, sigma_exp=sigma_mse, sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to

    num_particles = 100  # Define the number of particles
    sigma_mse = 1000.0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.0  # Define the value of sigma for the particles movement (dynamics)
    template_rect = {'x': int(320.8751), 'y': int(175.1776), 'w': int(102.5404), 'h': int(128.0504)}  # suggested template window (dict)

    run_particle_filter(ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "pres_debate.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-4-d-1.png'),
                            28: os.path.join(output_dir, 'ps7-4-d-2.png'),
                            94: os.path.join(output_dir, 'ps7-4-d-3.png'),
                            171: os.path.join(output_dir, 'ps7-4-d-4.png')
                        },  # frames to save, mapped to filenames, and 'template' if desired
                        num_particles=num_particles, sigma_exp=sigma_mse, sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to

    num_particles = 40  # Define the number of particles
    sigma_mse = 10.0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.0  # Define the value of sigma for the particles movement (dynamics)
    template_rect = {'x': int(320.8751), 'y': int(175.1776), 'w': int(102.5404), 'h': int(128.0504)}  # suggested template window (dict)

    run_particle_filter(ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "pres_debate.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-4-d-1.png'),
                            28: os.path.join(output_dir, 'ps7-4-d-2.png'),
                            94: os.path.join(output_dir, 'ps7-4-d-3.png'),
                            171: os.path.join(output_dir, 'ps7-4-d-4.png')
                        },  # frames to save, mapped to filenames, and 'template' if desired
                        num_particles=num_particles, sigma_exp=sigma_mse, sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to
    #'''

def part_5():
    num_particles = 150  # Define the number of particles
    sigma_md = 10.0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.0  # Define the value of sigma for the particles movement (dynamics)
    hist_bins_num = 6  # Define the number of bins
    #template_rect = {'x': 210, 'y': 70, 'w': 92, 'h': 234}  # Define the template window values
    template_rect = {'x': 230, 'y': 85, 'w': 70, 'h': 170}  # Define the template window values

    run_particle_filter(MDParticleFilter,
                        os.path.join(input_dir, "pedestrians.mp4"),
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-5-a-1.png'),
                            40: os.path.join(output_dir, 'ps7-5-a-2.png'),
                            100: os.path.join(output_dir, 'ps7-5-a-3.png'),
                            240: os.path.join(output_dir, 'ps7-5-a-4.png')
                        },
                        num_particles=num_particles, sigma_exp=sigma_md, sigma_dyn=sigma_dyn, hist_bins_num=hist_bins_num,
                        template_coords=template_rect)  # Add more if you need to

if __name__ == '__main__':
    #part_0a()
    part_1a()
    part_1b()
    part_2a()
    part_2b()
    part_3a()
    part_3b()
    part_4()
    part_5()
