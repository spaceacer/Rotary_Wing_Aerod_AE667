class Rotor:
    def __init__(self, n_blades, radius, root_cutout, chord_root, chord_tip, theta_root, theta_tip):
        self.n_blades = n_blades
        self.radius = radius
        self.root_cutout = root_cutout
        self.chord_root = chord_root
        self.chord_tip = chord_tip
        self.theta_root = theta_root
        self.theta_tip = theta_tip
        
    def get_chord(self, r):
        if self.radius == self.root_cutout:
            return self.chord_root
        return self.chord_root + (self.chord_tip - self.chord_root) * (r - self.root_cutout) / (self.radius - self.root_cutout)
        
    def get_twist(self, r):
        if self.radius == self.root_cutout:
            return self.theta_root
        return self.theta_root + (self.theta_tip - self.theta_root) * (r - self.root_cutout) / (self.radius - self.root_cutout)
        
    def get_cl_cd(self, alpha):
        # Linear aerodynamics from Knight and Hefner validation
        cl = 5.75 * alpha
        cd = 0.0113 + 1.25 * alpha**2
        return cl, cd
