import numpy as np
import scipy.optimize
from scipy.constants import e
from numpy import pi
from numpy import log as ln

class Measurement:
    """Class to store raw measurement data and perform some simple data cleaning"""
    default_skip_header = 4
    def __init__(self, filename, skip_header="default"):
        """Initializes Measurement

        Arguments
        ---------
        filename: (str) name of data file
        skip_header: (int or "default") how many header lines to skip before 
        reading data. If default, the default for the class will be used.
        """
        
        if skip_header == "default":
            skip_header = self.__class__.default_skip_header

        self._data = np.genfromtxt(filename, unpack=True, skip_header=skip_header)
        self.mask = None
        self._sorted = False

    @property
    def data(self):
        """array containing the cleaned data"""
        if self.mask is None:
            return self._data
        else:
            return self._data[:,self.mask]

    def sort(self):
        """sort the data by inceasing independent variable"""
        if self._independent_variable is not None:
            independent_variable = getattr(self, self._independent_variable)
            if not self._sorted:
                self._unsorted_mask = self.mask

            if self.mask is not None:
                if self.mask.dtype == bool:
                    self.mask = np.nonzero(self.mask)[0]
                self.mask = self.mask[np.argsort(independent_variable)]
            else:
                self.mask = np.argsort(independent_variable)
        else:
            raise ValueError("Can't sort without an independent variable")

    def unsort(self):
        """unsort the data"""
        if self._sorted:
            self.mask = self._unsorted_mask

def _get_column(name):
    return lambda s : s.data[s.variable_dict[name]]

def _set_column(name):
    def set_col(s, value):
        s._data[s.variable_dict[name], s.mask] = value
    return set_col

def define_measurement(column_dict, independent_variable = None,
        default_skip_header=4):
    """Class factory to make classes for storing raw data for different
    measurements.

    The class will have an attribute corresponding to each measured variable,
    which stores an array containing the measured values for that variable.

    Arguments
    ---------
    column_dictionary: dictionary with keys representing variable names and
        entries representing the corresponding columns
    independent_variable: name of the independent variable for sorting
        purposes (should match the key in column_dictionary)
    default_skip_header: default # of header lines to skip when reading data
        files
    """

    class Measurement_Class(Measurement):
        _independent_variable = independent_variable
        variable_list = list(column_dict.keys())
        variable_dict = column_dict

    for attribute in column_dict:
        if column_dict[attribute] is not None:
            getter = _get_column(attribute)
            setter = _set_column(attribute)
            setattr(Measurement_Class, attribute, property(fget=getter, fset=setter))

    Measurement_Class.default_skip_header = default_skip_header

    return Measurement_Class

Displex_RvsH_Data_Dict = {
        "Hxx" : 0, "Rxx" : 1, #T/Ohm
        "Hyy" : 2, "Ryy" : 3,
        "Hxy" : 4, "Rxy" : 5,
        "Hyx" : 6, "Ryx" : 7,
        "T" : 8, #temperature (K)
        "t" : 9, #time (min)
        }

PPMS_RvsH_dict = {
    "t": 0, #time (min)
    "T": 1, #temperature (K)
    "H": 2, #magnetic field (T)
    "theta": 3, #angle (degrees)
    "Rxx" : 6, "Ryy" : 9, #longitudinal resistance (Ohm)
    "Rxy" : 12, "Ryx" : 15, #transverse resistance (Ohm)
    }
for H_alias in ["Hxx", "Hyy", "Hxy", "Hyx"]:
    PPMS_RvsH_dict.update({H_alias: 2})
Raw_PPMS_RvsH_Data = define_measurement(
        PPMS_RvsH_dict, independent_variable = "H")

def interpolate(x, y):
    """linear interpolation of y(x) assuming x is monotonically increasing

    for x < x_min, sets y(x) = y(x_min) and behaves similarly for x > x_max

    Arguments
    _________
    x: (N,) array
    y: (N,) array

    Returns
    -------
    y_interpolated: function y(x) found by interpolation
    """
    def y_interpolated(x_new):
        return np.interp(x_new, x, y)
    return y_interpolated

linear = lambda x, m, b: m*x + b
linear_fit = lambda x, y: scipy.optimize.curve_fit(
    linear, x, y)

def local_extrema(y):
    """find all local extrema
    
    Arguments
    _________
    y: (N,) array

    Returns
    _______

    (N,) array of T/F values stating whether the corresponding index
    is an extremum.
    """
    positive_differences = (np.diff(y) > 0)
    extrema = (positive_differences[1:] ^ positive_differences[:-1]) \
        | (np.diff(y)[1:] == 0)
    return np.concatenate([[True], extrema, [True]])

class Hall_Measurement:
    """for storing/maniumpulating hall measurement data"""
    def __init__(self, filename, system, van_der_pauw=True, sort=False):
        """initialized Hall_Measurement

        Arguments
        _________
        filename: name of file containing measurement data
        system: "PPMS" or "displex"
        van_der_pauw: (T/F) whether the sample was measured with the van der
            pauw method (if T, multiplies Rxx by pi/ln(2))`
        sort: (T/F) whether to sort the (by increasing magnetic field)
        """

        self.system = system
        if system == "displex":
            self.raw = Raw_Displex_RvsH_Data(filename)
        elif system == "PPMS":
            self.raw = Raw_PPMS_RvsH_Data(filename)
            self.raw.H /= 1e4 #convert from Oe to T
        else:
            raise ValueError("Unkown system: {}. Available options are\
                    'displex' or 'PPMS'.".format(system))

        self.van_der_pauw = van_der_pauw
        self.sorted = sort

    @property
    def Hxx(self):
        """magnetic field measured at the same time as Rxx"""
        return self.raw.Hxx

    @property
    def Hxy(self):
        """magnetic field measured at the same time as Rxy"""
        return self.raw.Hxy

    @property
    def H(self):
        """magnetic field (only for PPMS)"""
        if self.system == "PPMS":
            return self.raw.H
        else:
            raise AttributeError("Displex measures H seperately for Rxx and \
                    Rxy. Use 'Hxx' or 'Hxy' instead.")

    @property 
    def sorted(self):
        """(T/F) whether to sort by increasing magnetic field"""
        return self._sorted
    
    @sorted.setter
    def sorted(self, value):
        self._sorted = value
        if value:
            self.raw.sort()
        else:
            self.raw.unsort()

    #WARNING: not robust to noise in the measurements
    def zero_at_Rmin(self):
        """sets H=0 to occur at minimum raw Rxx"""
        extrema = local_extrema(self.raw.Rxx + self.raw.Ryy)
        zero_index = np.where(abs(self.H) == np.min(abs(self.H[extrema])))[0][0]
        self.raw.H -= self.raw.H[zero_index]
    
    #longitudinal resistance (averaged over Rxx/Ryy) and symmetrized
    @property
    def Rxx(self):
        """symmetrized and averaged longitudinal resistance"""
        if not self.sorted:
            raise ValueError("Data must be sorted to interpolate")

        raw_Ryy = interpolate(self.raw.Hyy, self.raw.Ryy)
        R_averaged = interpolate(self.Hxx,
                                 (self.raw.Rxx + raw_Ryy(self.Hxx))/2)
        R_symmetrized = (R_averaged(self.Hxx) + R_averaged(-self.Hxx))/2
        if self.van_der_pauw:
            R_symmetrized *= pi/ln(2)
        return R_symmetrized
    
    #transverse resistance
    @property
    def Rxy(self):
        """antisymmetrized and averaged transverse resistance"""
        if not self.sorted:
            raise ValueError("Data must be sorted to interpolate")
        raw_Ryx = interpolate(self.raw.Hyx, self.raw.Ryx)
        R_averaged = interpolate(self.Hxy,
                                 (self.raw.Rxy + raw_Ryx(self.Hxy))/2)
        R_antisymmetrized = (R_averaged(self.Hxy) - R_averaged(-self.Hxy))/2
        return R_antisymmetrized
    
    @property
    def hall_coefficient(self):
        """hall coefficient R/H"""
        fitvals, fiterrs = linear_fit(self.Hxy, self.Rxy)
        return fitvals[0]
    
    @property
    def sheet_density(self):
        """sheet carrier density in cm^-2"""
        return 1e-4/(e*self.hall_coefficient)

#the PPMS RvsT data file has the same variable order as the RvsH 
RT_Measurement = define_measurement(PPMS_RvsH_dict, default_skip_header=3)

#The I/V measurement files store the data differently depending on whether 
#current or voltage is sourced.
_IV_Measurement = define_measurement({"I" : 0, "V" : 1}, default_skip_header=1)
class IV_Measurement(_IV_Measurement):
    """for storing/maniumpulating current vs voltage measurement data"""
    def __init__(self, filename, *args, **kwargs):
        """initializes IV_Measurement"""
        with open(filename) as file:
            if file.readline().startswith("Current"):
                self.source = "current"

            else:
                self.source = "voltage"
        super().__init__(filename, *args, **kwargs)

        if self.source == "voltage":
            self.variable_dict = {"I" : 1, "V" : 0} 
