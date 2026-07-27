// use std::{
//     collections::VecDeque,
//     sync::{Arc, Condvar, Mutex},
// };

// struct Shared<T> {
//     queue: Mutex<VecDeque<T>>,
//     available: Condvar,
// }

// pub struct Sender<T> {
//     inner: Arc<Shared<T>>,
// }

// impl<T> Sender<T> {
//     pub fn send(&self, value: T) {
//         let mut queue = self.inner.queue.lock().unwrap();
//         queue.push_back(value);
//         drop(queue);
//         self.inner.available.notify_one();
//     }
// }

// impl<T> Clone for Sender<T> {
//     fn clone(&self) -> Self {
//         Sender {
//             inner: Arc::clone(&self.inner),
//         }
//     }
// }

// pub struct Receiver<T> {
//     inner: Arc<Shared<T>>,
// }

// impl<T> Receiver<T> {
//     pub fn recv(&self) -> T {
//         let mut queue = self.inner.queue.lock().unwrap();
//         loop {
//             match queue.pop_front() {
//                 Some(value) => return value,
//                 None => {
//                     queue = self.inner.available.wait(queue).unwrap();
//                     return queue
//                         .pop_front()
//                         .expect("Queue should not be empty after wait");
//                 }
//             }
//         }
//     }
// }

// pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
//     let inner = Arc::new(Shared {
//         queue: Mutex::default(),
//         available: Condvar::default(),
//     });

//     (
//         Sender {
//             inner: inner.clone(),
//         },
//         Receiver {
//             inner: inner.clone(),
//         },
//     )
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_channel() {
//         let (sender, receiver) = channel();
//         sender.send(42);
//         assert_eq!(receiver.recv(), 42);
//     }

//     #[test]
//     fn test_threaded_channel() {
//         let (sender, receiver) = channel();
//         let handle = std::thread::spawn(move || {
//             sender.send(42);
//         });
//         assert_eq!(receiver.recv(), 42);
//         handle.join().unwrap();
//     }
// }
