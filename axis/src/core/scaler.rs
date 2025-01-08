#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    None,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
    Char,
    String,
    Bytes,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Scaler {
    Empty,
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Char(char),
    String(String),
    Bytes(Vec<u8>),
}

impl Scaler {
    pub fn is_empty(&self) -> bool {
        match self {
            Scaler::Empty => true,
            _ => false,
        }
    }

    pub fn is_numeric(&self) -> bool {
        match self {
            Scaler::I8(_)
            | Scaler::I16(_)
            | Scaler::I32(_)
            | Scaler::I64(_)
            | Scaler::U8(_)
            | Scaler::U16(_)
            | Scaler::U32(_)
            | Scaler::U64(_)
            | Scaler::F32(_)
            | Scaler::F64(_) => true,
            _ => false,
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Scaler::I8(_) => DataType::I8,
            Scaler::I16(_) => DataType::I16,
            Scaler::I32(_) => DataType::I32,
            Scaler::I64(_) => DataType::I64,
            Scaler::U8(_) => DataType::U8,
            Scaler::U16(_) => DataType::U16,
            Scaler::U32(_) => DataType::U32,
            Scaler::U64(_) => DataType::U64,
            Scaler::F32(_) => DataType::F32,
            Scaler::F64(_) => DataType::F64,
            Scaler::Bool(_) => DataType::Bool,
            Scaler::Char(_) => DataType::Char,
            Scaler::String(_) => DataType::String,
            Scaler::Bytes(_) => DataType::Bytes,
            _ => panic!("Invalid data type"),
        }
    }
}

impl From<i8> for Scaler {
    fn from(value: i8) -> Self {
        Scaler::I8(value)
    }
}

impl From<i16> for Scaler {
    fn from(value: i16) -> Self {
        Scaler::I16(value)
    }
}

impl From<i32> for Scaler {
    fn from(value: i32) -> Self {
        Scaler::I32(value)
    }
}

impl From<i64> for Scaler {
    fn from(value: i64) -> Self {
        Scaler::I64(value)
    }
}

impl From<u8> for Scaler {
    fn from(value: u8) -> Self {
        Scaler::U8(value)
    }
}

impl From<u16> for Scaler {
    fn from(value: u16) -> Self {
        Scaler::U16(value)
    }
}

impl From<u32> for Scaler {
    fn from(value: u32) -> Self {
        Scaler::U32(value)
    }
}

impl From<u64> for Scaler {
    fn from(value: u64) -> Self {
        Scaler::U64(value)
    }
}

impl From<f32> for Scaler {
    fn from(value: f32) -> Self {
        Scaler::F32(value)
    }
}

impl From<f64> for Scaler {
    fn from(value: f64) -> Self {
        Scaler::F64(value)
    }
}

impl From<bool> for Scaler {
    fn from(value: bool) -> Self {
        Scaler::Bool(value)
    }
}

impl From<char> for Scaler {
    fn from(value: char) -> Self {
        Scaler::Char(value)
    }
}

impl From<&str> for Scaler {
    fn from(value: &str) -> Self {
        Scaler::String(value.to_string())
    }
}

impl From<String> for Scaler {
    fn from(value: String) -> Self {
        Scaler::String(value)
    }
}

impl From<Vec<u8>> for Scaler {
    fn from(value: Vec<u8>) -> Self {
        Scaler::Bytes(value)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_is_empty() {
        let value = Scaler::Empty;
        assert_eq!(value.is_empty(), true);
    }

    #[test]
    fn test_from_i8() {
        let value: Scaler = 42i8.into();
        assert_eq!(value, Scaler::I8(42));
    }

    #[test]
    fn test_from_i16() {
        let value: Scaler = 42i16.into();
        assert_eq!(value, Scaler::I16(42));
    }

    #[test]
    fn test_from_i32() {
        let value: Scaler = 42i32.into();
        assert_eq!(value, Scaler::I32(42));
    }

    #[test]
    fn test_from_i64() {
        let value: Scaler = 42i64.into();
        assert_eq!(value, Scaler::I64(42));
    }

    #[test]
    fn test_from_u8() {
        let value: Scaler = 42u8.into();
        assert_eq!(value, Scaler::U8(42));
    }

    #[test]
    fn test_from_u16() {
        let value: Scaler = 42u16.into();
        assert_eq!(value, Scaler::U16(42));
    }

    #[test]
    fn test_from_u32() {
        let value: Scaler = 42u32.into();
        assert_eq!(value, Scaler::U32(42));
    }

    #[test]
    fn test_from_u64() {
        let value: Scaler = 42u64.into();
        assert_eq!(value, Scaler::U64(42));
    }

    #[test]
    fn test_from_f32() {
        let value: Scaler = 42.0f32.into();
        assert_eq!(value, Scaler::F32(42.0));
    }

    #[test]
    fn test_from_f64() {
        let value: Scaler = 42.0f64.into();
        assert_eq!(value, Scaler::F64(42.0));
    }

    #[test]
    fn test_from_bool() {
        let value: Scaler = true.into();
        assert_eq!(value, Scaler::Bool(true));
    }

    #[test]
    fn test_from_char() {
        let value: Scaler = 'a'.into();
        assert_eq!(value, Scaler::Char('a'));
    }
}
