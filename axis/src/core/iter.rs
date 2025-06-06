use std::collections::BTreeMap;

use super::Series;

pub struct FrameIterator<'a> {
    series: &'a BTreeMap<&'static str, Series>,
    order: &'a Vec<&'static str>,
    index: usize,
}

impl<'a> FrameIterator<'a> {
    pub fn new(series: &'a BTreeMap<&'static str, Series>, order: &'a Vec<&'static str>) -> Self {
        FrameIterator {
            series,
            order,
            index: 0,
        }
    }
}

impl<'a> Iterator for FrameIterator<'a> {
    type Item = &'a Series;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.order.len() {
            let name = self.order[self.index];
            self.index += 1;
            Some(&self.series[name])
        } else {
            None
        }
    }
}
